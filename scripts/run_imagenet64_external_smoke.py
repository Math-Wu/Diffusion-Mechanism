from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import fft_radial_band_energy, logsnr_bin_edges, radial_band_spec
from dm.imagenet64 import build_imagenet64_val_loader, load_imagenet64_pretrained_model
from dm.metrics import build_feature_extractor, feature_stats, frechet_distance
from dm.samplers.base import CountedModel, SamplerResult
from dm.samplers.ode import AB_COEFFS
from dm.schedules import CosineVPSchedule, LinearVPSchedule
from dm.utils import default_device, ensure_dir, set_seed


SOLVERS = ("ddim", "heun", "dpmpp", "unipc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small ImageNet64 pretrained P0/P2/p0_high validation smoke.")
    parser.add_argument("--data_npz", default="data/imagenet64/Imagenet64_val_npz/val_data.npz")
    parser.add_argument("--adm_checkpoint", default="checkpoints/guided-diffusion/64x64_diffusion.pt")
    parser.add_argument("--uvit_checkpoint", default="checkpoints/u-vit/imagenet64_uvit_large.pth")
    parser.add_argument("--architectures", nargs="+", default=["adm", "uvit"], choices=["adm", "uvit"])
    parser.add_argument("--solvers", nargs="+", default=["ddim", "heun", "dpmpp", "unipc"], choices=SOLVERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.2, 0.5])
    parser.add_argument("--output_dir", default="outputs/imagenet64_external_smoke")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--num_probe", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--time_bins", type=int, default=8)
    parser.add_argument("--freq_bands", type=int, default=4)
    parser.add_argument("--ref_substeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _macro_intervals(solver: str, nfe: int) -> int:
    return max(1, (nfe + 1) // 2) if solver == "heun" else max(1, nfe)


def _schedule_for_architecture(architecture: str):
    if architecture == "adm":
        return CosineVPSchedule(eps=1e-3)
    if architecture == "uvit":
        return LinearVPSchedule(eps=1e-4)
    raise ValueError(f"Unknown architecture: {architecture}")


def _checkpoint_for_architecture(args: argparse.Namespace, architecture: str) -> str:
    return args.adm_checkpoint if architecture == "adm" else args.uvit_checkpoint


def _high_band_slice(num_bands: int) -> slice:
    return slice(max(0, int(np.ceil(num_bands * 0.625))), num_bands)


def _normalize_profile(profile: np.ndarray, strength: float, max_density_ratio: float = 2.5) -> np.ndarray:
    profile = np.clip(np.asarray(profile, dtype=np.float64), a_min=0.0, a_max=None)
    if profile.size >= 3:
        padded = np.pad(profile, (1, 1), mode="edge")
        profile = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    if float(profile.sum()) <= 1e-12:
        profile = np.ones_like(profile)
    profile = profile / max(float(profile.mean()), 1e-12)
    profile = np.clip(profile, 1.0 / max_density_ratio, max_density_ratio)
    density = (1.0 - strength) + strength * profile
    return density / max(float(density.mean()), 1e-12)


def _time_grid_from_density(schedule, intervals: int, density: np.ndarray, device: torch.device) -> torch.Tensor:
    edges = logsnr_bin_edges(schedule, len(density), device=torch.device("cpu")).double().numpy()
    widths = np.diff(edges)
    mass = np.maximum(density, 1e-8) * widths
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    targets = np.linspace(0.0, float(cdf[-1]), intervals + 1)
    lambdas = np.interp(targets, cdf, edges)
    return schedule.inverse_log_snr(torch.tensor(lambdas, device=device, dtype=torch.float32))


class LabelConditionedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, labels: torch.Tensor):
        super().__init__()
        self.model = model
        self.labels = labels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t, self.labels[: x.shape[0]])


@torch.no_grad()
def _heun_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
    x_euler = x + dt * drift
    eps_next = model(x_euler, t_next)
    drift_next = schedule.drift(x_euler, t_next, eps_next)
    return x + 0.5 * dt * (drift + drift_next)


@torch.no_grad()
def _reference_heun(model, schedule, x: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, substeps: int) -> torch.Tensor:
    lambda_start = schedule.log_snr(t_start[0])
    lambda_end = schedule.log_snr(t_end[0])
    lambdas = torch.linspace(lambda_start, lambda_end, substeps + 1, device=x.device)
    times = schedule.inverse_log_snr(lambdas)
    current = x
    for index in range(substeps):
        current = _heun_step(
            model,
            schedule,
            current,
            _batch_time(times[index], x.shape[0]),
            _batch_time(times[index + 1], x.shape[0]),
        )
    return current


@torch.no_grad()
def _solver_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, solver: str, history: list[torch.Tensor]):
    if solver == "ddim":
        eps = model(x, t)
        x0 = schedule.eps_to_x0(x, t, eps)
        alpha_next, sigma_next = schedule.alpha_sigma(t_next)
        while alpha_next.ndim < x.ndim:
            alpha_next = alpha_next[..., None]
            sigma_next = sigma_next[..., None]
        return alpha_next * x0 + sigma_next * eps, history
    if solver == "heun":
        return _heun_step(model, schedule, x, t, t_next), history
    max_order = 2 if solver == "dpmpp" else 3
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
    history.insert(0, drift)
    order = min(max_order, len(history))
    update = torch.zeros_like(x)
    for coeff, old_drift in zip(AB_COEFFS[order], history):
        update = update + coeff * old_drift
    return x + dt * update, history[:max_order]


@torch.no_grad()
def _sample_grid(model, schedule, noise: torch.Tensor, labels: torch.Tensor, solver: str, nfe: int, times: torch.Tensor) -> SamplerResult:
    counted = CountedModel(LabelConditionedModel(model, labels))
    x = noise
    history: list[torch.Tensor] = []
    if solver == "heun":
        remaining = nfe
        for index in range(times.numel() - 1):
            t = _batch_time(times[index], x.shape[0])
            t_next = _batch_time(times[index + 1], x.shape[0])
            dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
            eps = counted(x, t)
            drift = schedule.drift(x, t, eps)
            remaining -= 1
            if remaining > 0:
                x_euler = x + dt * drift
                eps_next = counted(x_euler, t_next)
                drift_next = schedule.drift(x_euler, t_next, eps_next)
                x = x + 0.5 * dt * (drift + drift_next)
                remaining -= 1
            else:
                x = x + dt * drift
        return SamplerResult(samples=x, nfe=counted.nfe)
    for index in range(times.numel() - 1):
        t = _batch_time(times[index], x.shape[0])
        t_next = _batch_time(times[index + 1], x.shape[0])
        x, history = _solver_step(counted, schedule, x, t, t_next, solver, history)
    return SamplerResult(samples=x, nfe=counted.nfe)


def _limited_batches(loader, total: int, device: torch.device):
    seen = 0
    for x, y in loader:
        if seen >= total:
            break
        if seen + x.shape[0] > total:
            x = x[: total - seen]
            y = y[: total - seen]
        seen += x.shape[0]
        yield x.to(device), y.to(device)


@torch.no_grad()
def _difficulty_map(model, schedule, loader, total: int, time_bins: int, band_spec, device: torch.device) -> np.ndarray:
    times = schedule.inverse_log_snr(0.5 * (logsnr_bin_edges(schedule, time_bins, device)[:-1] + logsnr_bin_edges(schedule, time_bins, device)[1:]))
    sums = torch.zeros(time_bins, band_spec.masks.shape[0], device=device)
    counts = torch.zeros(time_bins, device=device)
    generator = torch.Generator(device=device).manual_seed(17)
    for x0, y in _limited_batches(loader, total, device):
        noise = torch.randn(x0.shape, device=device, generator=generator)
        for index, scalar_t in enumerate(times):
            t = _batch_time(scalar_t, x0.shape[0])
            x_t = schedule.q_sample(x0, t, noise)
            eps = model(x_t, t, y)
            x0_hat = schedule.eps_to_x0(x_t, t, eps)
            sums[index] += fft_radial_band_energy(x0_hat - x0, band_spec).sum(dim=0)
            counts[index] += x0.shape[0]
    return (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()


@torch.no_grad()
def _solver_error_map(model, schedule, loader, solver: str, nfe: int, total: int, time_bins: int, band_spec, ref_substeps: int, device: torch.device) -> np.ndarray:
    edges = logsnr_bin_edges(schedule, time_bins, device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lambda_step = (edges[-1] - edges[0]) / _macro_intervals(solver, nfe)
    sums = torch.zeros(time_bins, band_spec.masks.shape[0], device=device)
    counts = torch.zeros(time_bins, device=device)
    generator = torch.Generator(device=device).manual_seed(31)
    for x0, y in _limited_batches(loader, total, device):
        path_noise = torch.randn(x0.shape, device=device, generator=generator)
        conditioned = LabelConditionedModel(model, y)
        for bin_index, lambda_start in enumerate(centers):
            lambda_end = torch.minimum(lambda_start + lambda_step, edges[-1])
            t = _batch_time(schedule.inverse_log_snr(lambda_start), x0.shape[0])
            t_next = _batch_time(schedule.inverse_log_snr(lambda_end), x0.shape[0])
            x_start = schedule.q_sample(x0, t, path_noise)
            x_ref = _reference_heun(conditioned, schedule, x_start, t, t_next, ref_substeps)
            x_next, _ = _solver_step(conditioned, schedule, x_start, t, t_next, solver, history=[])
            sums[bin_index] += fft_radial_band_energy(x_next - x_ref, band_spec).sum(dim=0)
            counts[bin_index] += x0.shape[0]
    return (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()


@torch.no_grad()
def _p2_score(model, schedule, labels: torch.Tensor, noise: torch.Tensor, solver: str, nfe: int, band_spec, ref_substeps: int) -> dict[str, float]:
    conditioned = LabelConditionedModel(model, labels)
    times = schedule.time_grid(_macro_intervals(solver, nfe), noise.device)
    x_solver = noise.clone()
    x_ref = noise.clone()
    history: list[torch.Tensor] = []
    high = _high_band_slice(band_spec.masks.shape[0])
    x0_high = 0.0
    drift = 0.0
    count = 0
    for index in range(times.numel() - 1):
        t = _batch_time(times[index], noise.shape[0])
        t_next = _batch_time(times[index + 1], noise.shape[0])
        x_solver, history = _solver_step(conditioned, schedule, x_solver, t, t_next, solver, history)
        x_ref = _reference_heun(conditioned, schedule, x_ref, t, t_next, ref_substeps)
        eps_solver = conditioned(x_solver, t_next)
        eps_ref = conditioned(x_ref, t_next)
        x0_solver = schedule.eps_to_x0(x_solver, t_next, eps_solver)
        x0_ref = schedule.eps_to_x0(x_ref, t_next, eps_ref)
        x0_high += float(fft_radial_band_energy(x0_solver - x0_ref, band_spec)[:, high].sum().detach().cpu())
        drift += float((x_solver - x_ref).flatten(1).square().mean(dim=1).sqrt().sum().detach().cpu())
        count += noise.shape[0]
    return {"p2_x0_high_error": x0_high / max(count, 1), "p2_trajectory_drift": drift / max(count, 1)}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    loader = build_imagenet64_val_loader(args.data_npz, args.batch_size, num_workers=0, shuffle=False)
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    real_features = []
    for x, _y in _limited_batches(loader, args.num_samples, device):
        real_features.append(extractor(x).detach().float().cpu().numpy())
    real_stats = feature_stats(np.concatenate(real_features, axis=0)[: args.num_samples])

    p0_rows: list[dict[str, object]] = []
    p2_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []

    for architecture in args.architectures:
        model = load_imagenet64_pretrained_model(architecture, _checkpoint_for_architecture(args, architecture), device)
        schedule = _schedule_for_architecture(architecture)
        band_spec = radial_band_spec(64, 64, args.freq_bands, device)
        difficulty = _difficulty_map(model, schedule, loader, args.num_probe, args.time_bins, band_spec, device)
        difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)

        sample_loader = build_imagenet64_val_loader(args.data_npz, args.batch_size, num_workers=0, shuffle=False)
        labels_all = []
        for _x, y in itertools.islice(sample_loader, (args.num_samples + args.batch_size - 1) // args.batch_size):
            labels_all.append(y)
        labels_all = torch.cat(labels_all, dim=0)[: args.num_samples].to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise_all = torch.randn(args.num_samples, 3, 64, 64, device=device, generator=generator)

        for solver in args.solvers:
            for nfe in args.nfe:
                error_map = _solver_error_map(
                    model, schedule, loader, solver, nfe, args.num_probe, args.time_bins, band_spec, args.ref_substeps, device
                )
                p0_high = (difficulty_norm * error_map)[:, _high_band_slice(args.freq_bands)].sum(axis=1)
                p0_rows.append(
                    {
                        "architecture": architecture,
                        "solver": solver,
                        "nfe": nfe,
                        "difficulty_weighted_error": float((difficulty_norm * error_map).sum()),
                        "p0_high_sum": float(p0_high.sum()),
                        "total_error": float(error_map.sum()),
                    }
                )
                p2 = _p2_score(model, schedule, labels_all[: args.batch_size], noise_all[: args.batch_size], solver, nfe, band_spec, args.ref_substeps)
                p2_rows.append({"architecture": architecture, "solver": solver, "nfe": nfe, **p2})

                intervals = _macro_intervals(solver, nfe)
                densities = [("uniform", 0.0, np.ones(args.time_bins, dtype=np.float64))]
                densities.extend(("p0_high", strength, _normalize_profile(p0_high, strength)) for strength in args.strengths)
                for mode, strength, density in densities:
                    times = _time_grid_from_density(schedule, intervals, density, device)
                    features = []
                    total_nfe = 0
                    for start in range(0, args.num_samples, args.batch_size):
                        end = min(start + args.batch_size, args.num_samples)
                        result = _sample_grid(
                            model,
                            schedule,
                            noise_all[start:end],
                            labels_all[start:end],
                            solver,
                            nfe,
                            times,
                        )
                        total_nfe += result.nfe
                        features.append(extractor(result.samples).detach().float().cpu().numpy())
                    fid = frechet_distance(real_stats, feature_stats(np.concatenate(features, axis=0)[: args.num_samples]))
                    if mode == "uniform":
                        baseline_fid = fid
                    intervention_rows.append(
                        {
                            "architecture": architecture,
                            "solver": solver,
                            "nfe": nfe,
                            "mode": mode,
                            "strength": strength,
                            "fid": fid,
                            "delta_vs_uniform": fid - baseline_fid,
                            "num_samples": args.num_samples,
                            "num_probe": args.num_probe,
                            "total_model_calls": total_nfe,
                        }
                    )
                    print(
                        f"{architecture} {solver} nfe={nfe} {mode} strength={strength:g} "
                        f"fid={fid:.4f} delta={intervention_rows[-1]['delta_vs_uniform']:.4f}",
                        flush=True,
                    )
                    _write_csv(output_dir / "intervention_metrics.csv", intervention_rows)
                _write_csv(output_dir / "p0_scores.csv", p0_rows)
                _write_csv(output_dir / "p2_scores.csv", p2_rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(output_dir / "p0_scores.csv", flush=True)
    print(output_dir / "p2_scores.csv", flush=True)
    print(output_dir / "intervention_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
