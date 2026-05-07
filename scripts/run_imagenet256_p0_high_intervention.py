from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import fft_radial_band_energy, logsnr_bin_edges, radial_band_spec
from dm.eval_utils import load_or_create_noise_bank, noise_bank_id
from dm.imagenet256 import decode_latents, imagenet256_schedule, load_autoencoder_kl, load_imagenet256_model
from dm.metrics import build_feature_extractor, feature_stats, frechet_distance
from dm.samplers.base import SamplerResult
from dm.samplers.ode import AB_COEFFS
from dm.utils import default_device, ensure_dir, set_seed

from run_imagenet256_pretrained_sweep import (
    MODEL_SHAPES,
    SOLVERS,
    _default_noise_bank_path,
    _load_imagenet256_real_stats,
    _macro_intervals,
    _save_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet256 pretrained p0_high time-grid intervention using generated probe trajectories."
    )
    parser.add_argument("--model", choices=sorted(MODEL_SHAPES), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae_dir", help="Required for latent-space models.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--main_solver", choices=SOLVERS, required=True)
    parser.add_argument("--control_solvers", nargs="*", default=["heun", "dpmpp"], choices=SOLVERS)
    parser.add_argument("--main_strengths", nargs="+", type=float, default=[0.1, 0.2, 0.35, 0.5])
    parser.add_argument("--control_strengths", nargs="+", type=float, default=[0.35])
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_real_stats", type=int, default=10000)
    parser.add_argument("--num_probe", type=int, default=128)
    parser.add_argument("--probe_teacher_nfe", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--probe_batch_size", type=int, default=None)
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--time_bins", type=int, default=8)
    parser.add_argument("--freq_bands", type=int, default=4)
    parser.add_argument("--ref_substeps", type=int, default=16)
    parser.add_argument("--max_density_ratio", type=float, default=2.5)
    parser.add_argument("--smooth_profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--probe_noise_bank")
    parser.add_argument("--real_npz", default="data/imagenet256/ref_batches/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--baseline_metrics")
    parser.add_argument("--heun256_metrics")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    parser.add_argument("--grid_samples", type=int, default=64)
    parser.add_argument("--grid_nrow", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent_scale_factor", type=float, default=0.18215)
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _high_band_slice(num_bands: int) -> slice:
    start = max(0, int(np.ceil(num_bands * 0.625)))
    return slice(start, num_bands)


def _normalize_profile(profile: np.ndarray, strength: float, max_density_ratio: float, smooth: bool) -> np.ndarray:
    profile = np.clip(np.asarray(profile, dtype=np.float64), a_min=0.0, a_max=None)
    if smooth and profile.size >= 3:
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


@torch.no_grad()
def _ddim_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    eps = model(x, t, y)
    x0 = schedule.eps_to_x0(x, t, eps)
    alpha_next, sigma_next = schedule.alpha_sigma(t_next)
    while alpha_next.ndim < x.ndim:
        alpha_next = alpha_next[..., None]
        sigma_next = sigma_next[..., None]
    return alpha_next * x0 + sigma_next * eps


@torch.no_grad()
def _heun_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t, y)
    drift = schedule.drift(x, t, eps)
    x_euler = x + dt * drift
    eps_next = model(x_euler, t_next, y)
    drift_next = schedule.drift(x_euler, t_next, eps_next)
    return x + 0.5 * dt * (drift + drift_next)


@torch.no_grad()
def _ab_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    history: list[torch.Tensor],
    max_order: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t, y)
    drift = schedule.drift(x, t, eps)
    history.insert(0, drift)
    order = min(max_order, len(history))
    update = torch.zeros_like(x)
    for coeff, old_drift in zip(AB_COEFFS[order], history):
        update = update + coeff * old_drift
    return x + dt * update, history[:max_order]


@torch.no_grad()
def _reference_heun(
    model,
    schedule,
    x: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    y: torch.Tensor,
    substeps: int,
) -> torch.Tensor:
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
            y,
        )
    return current


@torch.no_grad()
def _solver_one_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    solver: str,
) -> torch.Tensor:
    if solver == "ddim":
        return _ddim_step(model, schedule, x, t, t_next, y)
    if solver == "heun":
        return _heun_step(model, schedule, x, t, t_next, y)
    max_order = 2 if solver == "dpmpp" else 3
    out, _history = _ab_step(model, schedule, x, t, t_next, y, [], max_order)
    return out


@torch.no_grad()
def _sample_custom_grid(
    model,
    schedule,
    noise: torch.Tensor,
    labels: torch.Tensor,
    solver: str,
    nfe: int,
    times: torch.Tensor,
) -> SamplerResult:
    x = noise
    calls = 0
    history: list[torch.Tensor] = []
    if solver == "heun":
        remaining = nfe
        for index in range(times.numel() - 1):
            t = _batch_time(times[index], x.shape[0])
            t_next = _batch_time(times[index + 1], x.shape[0])
            dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
            eps = model(x, t, labels)
            drift = schedule.drift(x, t, eps)
            remaining -= 1
            calls += 1
            if remaining > 0:
                x_euler = x + dt * drift
                eps_next = model(x_euler, t_next, labels)
                drift_next = schedule.drift(x_euler, t_next, eps_next)
                x = x + 0.5 * dt * (drift + drift_next)
                remaining -= 1
                calls += 1
            else:
                x = x + dt * drift
        return SamplerResult(samples=x, nfe=calls)

    for index in range(times.numel() - 1):
        t = _batch_time(times[index], x.shape[0])
        t_next = _batch_time(times[index + 1], x.shape[0])
        if solver == "ddim":
            x = _ddim_step(model, schedule, x, t, t_next, labels)
        elif solver == "dpmpp":
            x, history = _ab_step(model, schedule, x, t, t_next, labels, history, max_order=2)
        elif solver == "unipc":
            x, history = _ab_step(model, schedule, x, t, t_next, labels, history, max_order=3)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        calls += 1
    return SamplerResult(samples=x, nfe=calls)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_metric_lookup(path: str | None) -> dict[tuple[str, int], float]:
    lookup: dict[tuple[str, int], float] = {}
    if not path:
        return lookup
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lookup[(str(row["solver"]), int(row["nfe"]))] = float(row["fid"])
    return lookup


def _load_heun256_reference(path: str | None) -> float:
    if not path:
        return float("nan")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return float("nan")
    return float(rows[0]["fid"])


def _load_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_key(row: dict[str, object]) -> tuple[str, int, float]:
    return (str(row["solver"]), int(row["nfe"]), float(row["profile_strength"]))


@torch.no_grad()
def _generate_probe_x0(
    model,
    schedule,
    probe_noise: torch.Tensor,
    labels: torch.Tensor,
    nfe: int,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    times = schedule.time_grid(_macro_intervals("heun", nfe), device)
    chunks: list[torch.Tensor] = []
    for start in range(0, probe_noise.shape[0], batch_size):
        end = min(start + batch_size, probe_noise.shape[0])
        noise = probe_noise[start:end].to(device, non_blocking=True)
        y = labels[start:end].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            result = _sample_custom_grid(model, schedule, noise, y, "heun", nfe, times)
        chunks.append(result.samples.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def _load_or_create_probe_state(
    *,
    path: Path,
    model,
    schedule,
    probe_noise: torch.Tensor,
    labels: torch.Tensor,
    teacher_nfe: int,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        if int(payload["teacher_nfe"]) == teacher_nfe and tuple(payload["x0"].shape) == tuple(probe_noise.shape):
            return payload["x0"].float()
    x0 = _generate_probe_x0(model, schedule, probe_noise, labels, teacher_nfe, batch_size, device, amp_enabled)
    ensure_dir(path.parent)
    torch.save({"x0": x0, "teacher_nfe": teacher_nfe, "labels": labels.cpu()}, path)
    return x0


@torch.no_grad()
def _compute_difficulty_map(
    model,
    schedule,
    x0_probe: torch.Tensor,
    labels: torch.Tensor,
    time_bins: int,
    freq_bands: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    band_spec = radial_band_spec(x0_probe.shape[-2], x0_probe.shape[-1], freq_bands, device)
    centers = 0.5 * (
        logsnr_bin_edges(schedule, time_bins, device)[:-1] + logsnr_bin_edges(schedule, time_bins, device)[1:]
    )
    times = schedule.inverse_log_snr(centers)
    sums = torch.zeros(time_bins, freq_bands, device=device)
    counts = torch.zeros(time_bins, device=device)
    generator = torch.Generator(device=device).manual_seed(17)
    for start in range(0, x0_probe.shape[0], batch_size):
        end = min(start + batch_size, x0_probe.shape[0])
        x0 = x0_probe[start:end].to(device, non_blocking=True)
        y = labels[start:end].to(device, non_blocking=True)
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
def _compute_solver_error_map(
    model,
    schedule,
    x0_probe: torch.Tensor,
    labels: torch.Tensor,
    solver: str,
    nfe: int,
    time_bins: int,
    freq_bands: int,
    ref_substeps: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    band_spec = radial_band_spec(x0_probe.shape[-2], x0_probe.shape[-1], freq_bands, device)
    edges = logsnr_bin_edges(schedule, time_bins, device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lambda_step = (edges[-1] - edges[0]) / _macro_intervals(solver, nfe)
    sums = torch.zeros(time_bins, freq_bands, device=device)
    counts = torch.zeros(time_bins, device=device)
    generator = torch.Generator(device=device).manual_seed(31 + nfe + sum(ord(c) for c in solver))
    for start in range(0, x0_probe.shape[0], batch_size):
        end = min(start + batch_size, x0_probe.shape[0])
        x0 = x0_probe[start:end].to(device, non_blocking=True)
        y = labels[start:end].to(device, non_blocking=True)
        noise = torch.randn(x0.shape, device=device, generator=generator)
        for bin_index, lambda_start in enumerate(centers):
            lambda_end = torch.minimum(lambda_start + lambda_step, edges[-1])
            t = _batch_time(schedule.inverse_log_snr(lambda_start), x0.shape[0])
            t_next = _batch_time(schedule.inverse_log_snr(lambda_end), x0.shape[0])
            x_start = schedule.q_sample(x0, t, noise)
            x_ref = _reference_heun(model, schedule, x_start, t, t_next, y, ref_substeps)
            x_next = _solver_one_step(model, schedule, x_start, t, t_next, y, solver)
            sums[bin_index] += fft_radial_band_energy(x_next - x_ref, band_spec).sum(dim=0)
            counts[bin_index] += x0.shape[0]
    return (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()


def _profile_from_maps(difficulty: np.ndarray, solver_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
    component = difficulty_norm * solver_error
    p0_high = component[:, _high_band_slice(component.shape[1])].sum(axis=1)
    return p0_high, component


def _profile_cache_key(args: argparse.Namespace, solver: str, nfe: int) -> str:
    return (
        f"{args.model}_{solver}_nfe{nfe}_probe{args.num_probe}_teacher{args.probe_teacher_nfe}_"
        f"bins{args.time_bins}_bands{args.freq_bands}_ref{args.ref_substeps}"
    )


def _load_or_compute_profile(
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    model,
    schedule,
    x0_probe: torch.Tensor,
    labels: torch.Tensor,
    solver: str,
    nfe: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float], Path]:
    cache_path = cache_dir / f"{_profile_cache_key(args, solver, nfe)}.npz"
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        p0_high = np.asarray(payload["p0_high"], dtype=np.float64)
        component = np.asarray(payload["component"], dtype=np.float64)
    else:
        difficulty_cache = cache_dir / (
            f"{args.model}_difficulty_probe{args.num_probe}_teacher{args.probe_teacher_nfe}_"
            f"bins{args.time_bins}_bands{args.freq_bands}.npz"
        )
        if difficulty_cache.exists():
            difficulty = np.asarray(np.load(difficulty_cache, allow_pickle=False)["difficulty"], dtype=np.float64)
        else:
            print(f"difficulty_start model={args.model}", flush=True)
            difficulty = _compute_difficulty_map(
                model,
                schedule,
                x0_probe,
                labels,
                args.time_bins,
                args.freq_bands,
                batch_size,
                device,
            )
            np.savez_compressed(difficulty_cache, difficulty=difficulty)
            print(f"difficulty_done model={args.model} cache={difficulty_cache}", flush=True)
        print(f"solver_error_start model={args.model} solver={solver} nfe={nfe}", flush=True)
        solver_error = _compute_solver_error_map(
            model,
            schedule,
            x0_probe,
            labels,
            solver,
            nfe,
            args.time_bins,
            args.freq_bands,
            args.ref_substeps,
            batch_size,
            device,
        )
        p0_high, component = _profile_from_maps(difficulty, solver_error)
        np.savez_compressed(
            cache_path,
            difficulty=difficulty,
            solver_error=solver_error,
            component=component,
            p0_high=p0_high,
            solver=np.array(solver),
            nfe=np.array(nfe),
        )
        print(f"solver_error_done model={args.model} solver={solver} nfe={nfe} cache={cache_path}", flush=True)
    stats = {
        "p0_high_sum": float(p0_high.sum()),
        "p0_high_argmax_bin": float(p0_high.argmax()),
        "difficulty_weighted_error": float(component.sum()),
    }
    return p0_high, stats, cache_path


def _density_rows(args: argparse.Namespace, solver: str, nfe: int, strength: float, density: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "model": args.model,
            "solver": solver,
            "nfe": nfe,
            "intervention_mode": "p0_high",
            "profile_strength": strength,
            "bin": index,
            "density": float(value),
        }
        for index, value in enumerate(density)
    ]


def _gap_rows(args: argparse.Namespace, solver: str, nfe: int, strength: float, schedule, times: torch.Tensor) -> list[dict[str, object]]:
    lambdas = schedule.log_snr(times).detach().float().cpu().numpy()
    gaps = np.diff(lambdas)
    return [
        {
            "model": args.model,
            "solver": solver,
            "nfe": nfe,
            "intervention_mode": "p0_high",
            "profile_strength": strength,
            "interval": index,
            "t_start": float(times[index].detach().cpu()),
            "t_end": float(times[index + 1].detach().cpu()),
            "logsnr_start": float(lambdas[index]),
            "logsnr_end": float(lambdas[index + 1]),
            "logsnr_gap": float(gaps[index]),
        }
        for index in range(len(gaps))
    ]


def _strength_plan(args: argparse.Namespace) -> list[tuple[str, list[float]]]:
    seen: set[str] = set()
    plan: list[tuple[str, list[float]]] = []
    for solver in [args.main_solver, *args.control_solvers]:
        if solver in seen:
            continue
        seen.add(solver)
        strengths = args.main_strengths if solver == args.main_solver else args.control_strengths
        plan.append((solver, strengths))
    return plan


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    grid_dir = ensure_dir(output_dir / "sample_grids")
    cache_dir = ensure_dir(output_dir / "profile_cache")
    probe_batch_size = args.probe_batch_size or args.batch_size
    amp_enabled = bool(args.amp and device.type == "cuda")
    print(f"imagenet256_p0_high_start model={args.model} device={device} output_dir={output_dir}", flush=True)

    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    real_stats = _load_imagenet256_real_stats(
        npz_path=args.real_npz,
        cache_path=args.real_stats_cache or output_dir / f"imagenet256_val_{args.feature_backend}_n{args.num_real_stats}.npz",
        extractor=extractor,
        device=device,
        num_samples=args.num_real_stats,
        batch_size=args.real_batch_size,
        feature_backend=args.feature_backend,
    )
    schedule = imagenet256_schedule()
    model = load_imagenet256_model(args.model, args.checkpoint, device)
    vae = None
    if args.model != "adm256":
        if not args.vae_dir:
            raise ValueError("--vae_dir is required for latent-space ImageNet256 models")
        vae = load_autoencoder_kl(args.vae_dir, device)

    shape = MODEL_SHAPES[args.model]
    noise_path = args.noise_bank or _default_noise_bank_path(args.model, args.seed, args.num_samples, shape)
    noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=shape, seed=args.seed)
    probe_noise_path = args.probe_noise_bank or _default_noise_bank_path(
        args.model, args.seed + 91001, args.num_probe, shape
    )
    probe_noise = load_or_create_noise_bank(probe_noise_path, num_samples=args.num_probe, shape=shape, seed=args.seed + 91001)
    noise_id = noise_bank_id(noise_path)
    labels = (torch.arange(args.num_samples, dtype=torch.long) % 1000).contiguous()
    probe_labels = (torch.arange(args.num_probe, dtype=torch.long) % 1000).contiguous()
    probe_x0_path = cache_dir / (
        f"{args.model}_probe_x0_n{args.num_probe}_teacher_heun{args.probe_teacher_nfe}_{'x'.join(map(str, shape))}.pt"
    )
    print(f"probe_x0_start model={args.model} path={probe_x0_path}", flush=True)
    x0_probe = _load_or_create_probe_state(
        path=probe_x0_path,
        model=model,
        schedule=schedule,
        probe_noise=probe_noise,
        labels=probe_labels,
        teacher_nfe=args.probe_teacher_nfe,
        batch_size=probe_batch_size,
        device=device,
        amp_enabled=amp_enabled,
    )
    print(f"probe_x0_ready model={args.model} shape={tuple(x0_probe.shape)}", flush=True)

    baseline_lookup = _load_metric_lookup(args.baseline_metrics)
    heun256_fid = _load_heun256_reference(args.heun256_metrics)
    metrics_path = output_dir / "p0_high_metrics.csv"
    rows = _load_existing_rows(metrics_path)
    completed = {_row_key(row) for row in rows}
    density_rows: list[dict[str, object]] = _load_existing_rows(output_dir / "timegrid_density.csv")
    gap_rows: list[dict[str, object]] = _load_existing_rows(output_dir / "timegrid_gaps.csv")

    for solver, strengths in _strength_plan(args):
        for nfe in args.nfe:
            p0_high, profile_stats, profile_cache = _load_or_compute_profile(
                args=args,
                cache_dir=cache_dir,
                model=model,
                schedule=schedule,
                x0_probe=x0_probe,
                labels=probe_labels,
                solver=solver,
                nfe=nfe,
                device=device,
                batch_size=probe_batch_size,
            )
            intervals = _macro_intervals(solver, nfe)
            for strength in strengths:
                key = (solver, nfe, float(strength))
                if key in completed:
                    print(f"skip_completed model={args.model} solver={solver} nfe={nfe} strength={strength:g}", flush=True)
                    continue
                density = _normalize_profile(p0_high, strength, args.max_density_ratio, args.smooth_profile)
                times = _time_grid_from_density(schedule, intervals, density, device)
                density_rows.extend(_density_rows(args, solver, nfe, strength, density))
                gap_rows.extend(_gap_rows(args, solver, nfe, strength, schedule, times))
                features: list[np.ndarray] = []
                grid_images: list[torch.Tensor] = []
                total_calls = 0
                total_finite = 0
                total_pixels = 0
                pixel_sum = 0.0
                pixel_sq_sum = 0.0
                pixel_min = float("inf")
                pixel_max = float("-inf")
                sampling_runtime = 0.0
                start_time = time.time()
                print(
                    f"intervention_start model={args.model} solver={solver} nfe={nfe} strength={strength:g}",
                    flush=True,
                )
                for start in range(0, args.num_samples, args.batch_size):
                    end = min(start + args.batch_size, args.num_samples)
                    noise = noise_bank[start:end].to(device, non_blocking=True)
                    y = labels[start:end].to(device, non_blocking=True)
                    sample_start = time.time()
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        result = _sample_custom_grid(model, schedule, noise, y, solver, nfe, times)
                        samples = result.samples
                        if vae is not None:
                            samples = decode_latents(vae, samples, scale_factor=args.latent_scale_factor)
                    sampling_runtime += time.time() - sample_start
                    samples = samples.detach().float()
                    features.append(extractor(samples).detach().float().cpu().numpy())
                    samples_cpu = samples.cpu()
                    total_calls += result.nfe
                    total_finite += int(torch.isfinite(samples_cpu).sum().item())
                    total_pixels += int(samples_cpu.numel())
                    pixel_sum += float(samples_cpu.sum().item())
                    pixel_sq_sum += float(samples_cpu.square().sum().item())
                    pixel_min = min(pixel_min, float(samples_cpu.min().item()))
                    pixel_max = max(pixel_max, float(samples_cpu.max().item()))
                    remaining_grid = args.grid_samples - sum(item.shape[0] for item in grid_images)
                    if remaining_grid > 0:
                        grid_images.append(samples_cpu[:remaining_grid])
                elapsed = time.time() - start_time
                fid = frechet_distance(real_stats, feature_stats(np.concatenate(features, axis=0)[: args.num_samples]))
                baseline_fid = baseline_lookup.get((solver, nfe), float("nan"))
                mean = pixel_sum / max(total_pixels, 1)
                variance = max(pixel_sq_sum / max(total_pixels, 1) - mean * mean, 0.0)
                finite_fraction = total_finite / max(total_pixels, 1)
                grid_path = grid_dir / f"{args.model}_{solver}_nfe{nfe}_p0_high_s{strength:g}.png"
                if grid_images:
                    _save_grid(grid_path, torch.cat(grid_images, dim=0), args.grid_nrow)
                row = {
                    "architecture": args.model,
                    "model": args.model,
                    "solver": solver,
                    "nfe": nfe,
                    "intervention_mode": "p0_high",
                    "profile_strength": strength,
                    "fid": fid,
                    "uniform_fid": baseline_fid,
                    "delta_vs_uniform": fid - baseline_fid if np.isfinite(baseline_fid) else float("nan"),
                    "heun256_reference_fid": heun256_fid,
                    "delta_fid_to_heun256": fid - heun256_fid if np.isfinite(heun256_fid) else float("nan"),
                    "num_samples": args.num_samples,
                    "num_real_stats": args.num_real_stats,
                    "num_probe": args.num_probe,
                    "probe_teacher_nfe": args.probe_teacher_nfe,
                    "time_bins": args.time_bins,
                    "freq_bands": args.freq_bands,
                    "ref_substeps": args.ref_substeps,
                    "batch_size": args.batch_size,
                    "probe_batch_size": probe_batch_size,
                    "wall_clock_sec": sampling_runtime,
                    "total_elapsed_sec": elapsed,
                    "sec_per_sample": sampling_runtime / max(args.num_samples, 1),
                    "nfe_per_sample": nfe,
                    "model_call_batches": total_calls,
                    "finite_fraction": finite_fraction,
                    "pixel_mean": mean,
                    "pixel_std": variance**0.5,
                    "pixel_min": pixel_min,
                    "pixel_max": pixel_max,
                    "density_min": float(density.min()),
                    "density_max": float(density.max()),
                    "density_argmax_bin": int(density.argmax()),
                    "p0_high_sum": profile_stats["p0_high_sum"],
                    "p0_high_argmax_bin": int(profile_stats["p0_high_argmax_bin"]),
                    "difficulty_weighted_error": profile_stats["difficulty_weighted_error"],
                    "profile_cache": str(profile_cache),
                    "profile_source": f"generated_probe_heun{args.probe_teacher_nfe}_state_space",
                    "checkpoint": args.checkpoint,
                    "vae_dir": args.vae_dir or "",
                    "seed": args.seed,
                    "noise_bank_id": noise_id,
                    "noise_bank": str(noise_path),
                    "probe_noise_bank": str(probe_noise_path),
                    "real_npz": str(args.real_npz),
                    "real_stats_cache": str(args.real_stats_cache or output_dir / f"imagenet256_val_{args.feature_backend}_n{args.num_real_stats}.npz"),
                    "baseline_metrics": args.baseline_metrics or "",
                    "heun256_metrics": args.heun256_metrics or "",
                    "feature_backend": args.feature_backend,
                    "grid_path": str(grid_path),
                    "metric_note": "imagenet256_p0_high_intervention_torchvision_inception_proxy_fid",
                }
                rows.append(row)
                completed.add(key)
                _write_csv(metrics_path, rows)
                _write_csv(output_dir / "timegrid_density.csv", density_rows)
                _write_csv(output_dir / "timegrid_gaps.csv", gap_rows)
                print(
                    f"intervention_done model={args.model} solver={solver} nfe={nfe} strength={strength:g} "
                    f"fid={fid:.4f} delta_uniform={row['delta_vs_uniform']:.4f} "
                    f"delta_heun256={row['delta_fid_to_heun256']:.4f} sample_sec={sampling_runtime:.2f}",
                    flush=True,
                )
    print(metrics_path, flush=True)


if __name__ == "__main__":
    main()
