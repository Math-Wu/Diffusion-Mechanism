from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dm_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import (
    fft_radial_band_energy,
    frequency_band_names,
    high_band_slice,
    logsnr_bin_edges,
    low_mid_band_slice,
    radial_band_spec,
    radial_power_profile,
)
from dm.cifar_public import build_public_cifar_model, public_cifar_schedule
from dm.data import build_cifar10_loaders
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.imagenet64 import build_imagenet64_val_loader, load_imagenet64_pretrained_model
from dm.schedules import CosineVPSchedule, LinearVPSchedule
from dm.utils import default_device, ensure_dir, set_seed
from run_imagenet64_external_smoke import _normalize_profile, _time_grid_from_density


DDIM_NFES = [4, 6, 8]
HEUN_NFES = [4, 6, 7, 8]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    dataset: str
    model_name: str
    solver: str
    nfes: tuple[int, ...]
    image_size: int


@dataclass
class CaseRuntime:
    spec: CaseSpec
    model: nn.Module
    schedule: CosineVPSchedule | LinearVPSchedule
    loader: Iterable
    batch_size: int
    checkpoint: str


CASE_SPECS: dict[str, CaseSpec] = {
    "cifar_self_uvit_ddim": CaseSpec("cifar_self_uvit_ddim", "cifar10", "self_uvit", "ddim", tuple(DDIM_NFES), 32),
    "cifar_public_uvit_ddim": CaseSpec("cifar_public_uvit_ddim", "cifar10", "public_uvit", "ddim", tuple(DDIM_NFES), 32),
    "cifar_public_ddpm_unet_ddim": CaseSpec(
        "cifar_public_ddpm_unet_ddim", "cifar10", "public_ddpm_unet", "ddim", tuple(DDIM_NFES), 32
    ),
    "imagenet64_adm_ddim": CaseSpec("imagenet64_adm_ddim", "imagenet64", "adm", "ddim", tuple(DDIM_NFES), 64),
    "imagenet64_uvit_ddim": CaseSpec("imagenet64_uvit_ddim", "imagenet64", "uvit", "ddim", tuple(DDIM_NFES), 64),
    "imagenet64_uvit_heun": CaseSpec("imagenet64_uvit_heun", "imagenet64", "uvit", "heun", tuple(HEUN_NFES), 64),
}


class UnconditionalLabelAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        del y
        return self.model(x, t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case-control frequency diagnostics for p0_high sampler mechanisms.")
    parser.add_argument("--output_dir", default="outputs/frequency_diagnosis_minimal_20260503")
    parser.add_argument("--cases", nargs="+", default=list(CASE_SPECS), choices=sorted(CASE_SPECS))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["dataset_profile", "band_error", "spectral_transfer", "grid_sacrifice"],
        choices=["dataset_profile", "band_error", "spectral_transfer", "grid_sacrifice"],
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split_seed", type=int, default=20260423)
    parser.add_argument("--imagenet64_npz", default="data/imagenet64/Imagenet64_val_npz/val_data.npz")
    parser.add_argument("--self_uvit_config", default="configs/cifar_medium/uvit.yaml")
    parser.add_argument("--self_uvit_run_dir", default="outputs/cifar_uvit_s0_medium_d10_h640")
    parser.add_argument("--self_uvit_checkpoint_name", default="last.pt")
    parser.add_argument("--public_ddpm_unet_checkpoint", default="checkpoints/ddpm-cifar10-32")
    parser.add_argument("--public_uvit_checkpoint", default="checkpoints/u-vit-cifar10/cifar10_uvit_small.pth")
    parser.add_argument("--adm_checkpoint", default="checkpoints/guided-diffusion/64x64_diffusion.pt")
    parser.add_argument("--imagenet64_uvit_checkpoint", default="checkpoints/u-vit/imagenet64_uvit_large.pth")
    parser.add_argument("--num_probe", type=int, default=256)
    parser.add_argument("--num_dataset_profile", type=int, default=2048)
    parser.add_argument("--batch_size_cifar", type=int, default=64)
    parser.add_argument("--batch_size_imagenet64", type=int, default=8)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=4)
    parser.add_argument("--radial_bins", type=int, default=32)
    parser.add_argument("--ref_substeps", type=int, default=16)
    parser.add_argument("--p0_high_strength", type=float, default=0.5)
    parser.add_argument("--max_density_ratio", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip_missing", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _macro_intervals(solver: str, nfe: int) -> int:
    return max(1, (nfe + 1) // 2) if solver == "heun" else max(1, nfe)


def _heun_interval_is_full_corrector(nfe: int, interval_index: int) -> bool:
    return interval_index < nfe // 2


def _predict_eps(model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return model(x, t, y)


@torch.no_grad()
def _ddim_step(model: nn.Module, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    eps = _predict_eps(model, x, t, y)
    x0 = schedule.eps_to_x0(x, t, eps)
    alpha_next, sigma_next = schedule.alpha_sigma(t_next)
    while alpha_next.ndim < x.ndim:
        alpha_next = alpha_next[..., None]
        sigma_next = sigma_next[..., None]
    return alpha_next * x0 + sigma_next * eps


@torch.no_grad()
def _euler_step(model: nn.Module, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = _predict_eps(model, x, t, y)
    drift = schedule.drift(x, t, eps)
    return x + dt * drift


@torch.no_grad()
def _heun_step(
    model: nn.Module,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    *,
    full_corrector: bool = True,
) -> torch.Tensor:
    if not full_corrector:
        return _euler_step(model, schedule, x, t, t_next, y)
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = _predict_eps(model, x, t, y)
    drift = schedule.drift(x, t, eps)
    x_euler = x + dt * drift
    eps_next = _predict_eps(model, x_euler, t_next, y)
    drift_next = schedule.drift(x_euler, t_next, eps_next)
    return x + 0.5 * dt * (drift + drift_next)


@torch.no_grad()
def _reference_heun(
    model: nn.Module,
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
            full_corrector=True,
        )
    return current


@torch.no_grad()
def _solver_step(
    model: nn.Module,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    solver: str,
    *,
    heun_full_corrector: bool = True,
) -> torch.Tensor:
    if solver == "ddim":
        return _ddim_step(model, schedule, x, t, t_next, y)
    if solver == "heun":
        return _heun_step(model, schedule, x, t, t_next, y, full_corrector=heun_full_corrector)
    raise ValueError(f"Unsupported frequency diagnosis solver: {solver}")


def _cifar_config(args: argparse.Namespace, batch_size: int) -> dict:
    return {
        "data": {
            "root": args.data_root,
            "split_seed": args.split_seed,
            "train_size": 45000,
            "val_size": 5000,
            "num_workers": 2,
        },
        "training": {"batch_size": batch_size},
    }


def _limited_batches(loader: Iterable, total: int, device: torch.device):
    seen = 0
    for batch in loader:
        if seen >= total:
            break
        x, y = batch
        if seen + x.shape[0] > total:
            x = x[: total - seen]
            y = y[: total - seen]
        seen += x.shape[0]
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _build_loader_for_dataset(args: argparse.Namespace, dataset: str, batch_size: int):
    if dataset == "cifar10":
        _, val_loader = build_cifar10_loaders(_cifar_config(args, batch_size), download=False)
        return val_loader
    if dataset == "imagenet64":
        return build_imagenet64_val_loader(args.imagenet64_npz, batch_size, num_workers=0, shuffle=False)
    raise ValueError(f"Unknown dataset: {dataset}")


def _schedule_for_imagenet64_model(model_name: str):
    if model_name == "adm":
        return CosineVPSchedule(eps=1e-3)
    if model_name == "uvit":
        return LinearVPSchedule(eps=1e-4)
    raise ValueError(f"Unknown ImageNet64 model: {model_name}")


def _case_batch_size(args: argparse.Namespace, spec: CaseSpec) -> int:
    return args.batch_size_imagenet64 if spec.dataset == "imagenet64" else args.batch_size_cifar


def _load_case_runtime(args: argparse.Namespace, spec: CaseSpec, device: torch.device) -> CaseRuntime:
    batch_size = _case_batch_size(args, spec)
    loader = _build_loader_for_dataset(args, spec.dataset, batch_size)
    checkpoint = ""
    if spec.model_name == "self_uvit":
        ckpt = checkpoint_path_for_run(args.self_uvit_run_dir, name=args.self_uvit_checkpoint_name)
        model, config, _checkpoint_payload = load_model_from_checkpoint(args.self_uvit_config, ckpt, device, use_ema=True)
        schedule = build_schedule(config)
        checkpoint = str(ckpt)
        wrapped_model = UnconditionalLabelAdapter(model).to(device).eval()
    elif spec.model_name in {"public_uvit", "public_ddpm_unet"}:
        checkpoint = args.public_uvit_checkpoint if spec.model_name == "public_uvit" else args.public_ddpm_unet_checkpoint
        wrapped_model = build_public_cifar_model(spec.model_name, checkpoint, device)
        schedule = public_cifar_schedule()
    elif spec.dataset == "imagenet64":
        checkpoint = args.adm_checkpoint if spec.model_name == "adm" else args.imagenet64_uvit_checkpoint
        wrapped_model = load_imagenet64_pretrained_model(spec.model_name, checkpoint, device)
        schedule = _schedule_for_imagenet64_model(spec.model_name)
    else:
        raise ValueError(f"Unsupported case: {spec.case_id}")
    return CaseRuntime(spec=spec, model=wrapped_model, schedule=schedule, loader=loader, batch_size=batch_size, checkpoint=checkpoint)


def _fit_loglog_slope(radius: np.ndarray, power: np.ndarray) -> float:
    mask = (radius > 0.0) & (power > 0.0)
    if int(mask.sum()) < 3:
        return float("nan")
    x = np.log(radius[mask])
    y = np.log(power[mask])
    return float(np.polyfit(x, y, deg=1)[0])


@torch.no_grad()
def compute_dataset_profiles(args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for dataset, image_size, batch_size in [
        ("cifar10", 32, args.batch_size_cifar),
        ("imagenet64", 64, args.batch_size_imagenet64),
    ]:
        loader = _build_loader_for_dataset(args, dataset, batch_size)
        energy_sum = torch.zeros(args.radial_bins, device=device)
        power_sum = torch.zeros(args.radial_bins, device=device)
        coeff_counts = None
        sample_count = 0
        for x, _y in _limited_batches(loader, args.num_dataset_profile, device):
            profile = radial_power_profile(x, args.radial_bins)
            energy_sum += profile.energy.sum(dim=0)
            power_sum += profile.mean_power.sum(dim=0)
            coeff_counts = profile.counts.detach().cpu().numpy()
            sample_count += x.shape[0]
            edges = profile.edges.detach().cpu().numpy()
        if sample_count == 0:
            raise RuntimeError(f"No samples available for dataset profile: {dataset}")
        mean_energy = (energy_sum / sample_count).detach().cpu().numpy()
        mean_power = (power_sum / sample_count).detach().cpu().numpy()
        norm_power = mean_power / max(float(mean_power.sum()), 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        high_start = int(math.ceil(args.radial_bins * 0.625))
        high_energy_ratio = float(mean_energy[high_start:].sum() / max(float(mean_energy.sum()), 1e-12))
        slope = _fit_loglog_slope(centers[1:], mean_power[1:])
        summary_rows.append(
            {
                "dataset": dataset,
                "image_size": image_size,
                "num_samples": sample_count,
                "radial_bins": args.radial_bins,
                "high_start_bin": high_start,
                "high_energy_ratio": high_energy_ratio,
                "loglog_power_slope": slope,
            }
        )
        for index in range(args.radial_bins):
            rows.append(
                {
                    "dataset": dataset,
                    "image_size": image_size,
                    "num_samples": sample_count,
                    "radial_bin": index,
                    "radius_left": float(edges[index]),
                    "radius_right": float(edges[index + 1]),
                    "radius_center": float(centers[index]),
                    "mean_energy": float(mean_energy[index]),
                    "mean_power": float(mean_power[index]),
                    "normalized_power": float(norm_power[index]),
                    "fft_coeff_count_per_channel": float(coeff_counts[index] / 3.0),
                }
            )
        print(f"dataset_profile_done dataset={dataset} samples={sample_count}", flush=True)
    _write_csv(output_dir / "dataset_frequency_profile.csv", rows)
    _write_csv(output_dir / "dataset_frequency_summary.csv", summary_rows)
    plot_dataset_profiles(rows, summary_rows, output_dir / "figures")


@torch.no_grad()
def compute_difficulty_map(runtime: CaseRuntime, args: argparse.Namespace, band_spec, device: torch.device) -> np.ndarray:
    times = 0.5 * (
        logsnr_bin_edges(runtime.schedule, args.time_bins, device)[:-1]
        + logsnr_bin_edges(runtime.schedule, args.time_bins, device)[1:]
    )
    times = runtime.schedule.inverse_log_snr(times)
    sums = torch.zeros(args.time_bins, args.freq_bands, device=device)
    counts = torch.zeros(args.time_bins, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    for x0, y in _limited_batches(runtime.loader, args.num_probe, device):
        noise = torch.randn(x0.shape, device=device, generator=generator)
        for index, scalar_t in enumerate(times):
            t = _batch_time(scalar_t, x0.shape[0])
            x_t = runtime.schedule.q_sample(x0, t, noise)
            eps = _predict_eps(runtime.model, x_t, t, y)
            x0_hat = runtime.schedule.eps_to_x0(x_t, t, eps)
            sums[index] += fft_radial_band_energy(x0_hat - x0, band_spec).sum(dim=0)
            counts[index] += x0.shape[0]
    return (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()


@torch.no_grad()
def compute_bandwise_local_error(
    runtime: CaseRuntime,
    args: argparse.Namespace,
    nfe: int,
    band_spec,
    device: torch.device,
) -> dict[str, np.ndarray]:
    edges = logsnr_bin_edges(runtime.schedule, args.time_bins, device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    intervals = _macro_intervals(runtime.spec.solver, nfe)
    lambda_step = (edges[-1] - edges[0]) / intervals
    error_sums = torch.zeros(args.time_bins, args.freq_bands, device=device)
    ref_sums = torch.zeros(args.time_bins, args.freq_bands, device=device)
    start_sums = torch.zeros(args.time_bins, args.freq_bands, device=device)
    counts = torch.zeros(args.time_bins, device=device)
    interval_indices = torch.clamp(torch.floor((centers - edges[0]) / lambda_step), 0, intervals - 1).long()
    heun_full = torch.tensor(
        [_heun_interval_is_full_corrector(nfe, int(index.item())) for index in interval_indices],
        device=device,
        dtype=torch.bool,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 31 + nfe)
    for x0, y in _limited_batches(runtime.loader, args.num_probe, device):
        path_noise = torch.randn(x0.shape, device=device, generator=generator)
        for bin_index, lambda_start in enumerate(centers):
            lambda_end = torch.minimum(lambda_start + lambda_step, edges[-1])
            t = _batch_time(runtime.schedule.inverse_log_snr(lambda_start), x0.shape[0])
            t_next = _batch_time(runtime.schedule.inverse_log_snr(lambda_end), x0.shape[0])
            x_start = runtime.schedule.q_sample(x0, t, path_noise)
            x_ref = _reference_heun(runtime.model, runtime.schedule, x_start, t, t_next, y, args.ref_substeps)
            x_next = _solver_step(
                runtime.model,
                runtime.schedule,
                x_start,
                t,
                t_next,
                y,
                runtime.spec.solver,
                heun_full_corrector=bool(heun_full[bin_index].item()),
            )
            error_sums[bin_index] += fft_radial_band_energy(x_next - x_ref, band_spec).sum(dim=0)
            ref_sums[bin_index] += fft_radial_band_energy(x_ref, band_spec).sum(dim=0)
            start_sums[bin_index] += fft_radial_band_energy(x_start, band_spec).sum(dim=0)
            counts[bin_index] += x0.shape[0]
    abs_error = error_sums / counts.clamp_min(1.0)[:, None]
    ref_energy = ref_sums / counts.clamp_min(1.0)[:, None]
    start_energy = start_sums / counts.clamp_min(1.0)[:, None]
    norm_error = abs_error / ref_energy.clamp_min(1e-12)
    return {
        "abs_error": abs_error.detach().cpu().numpy(),
        "norm_error": norm_error.detach().cpu().numpy(),
        "ref_energy": ref_energy.detach().cpu().numpy(),
        "start_energy": start_energy.detach().cpu().numpy(),
        "interval_indices": interval_indices.detach().cpu().numpy(),
        "heun_full_corrector": heun_full.detach().cpu().numpy().astype(np.int64),
    }


def _profile_from_maps(difficulty: np.ndarray, error_map: np.ndarray, freq_bands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
    component = difficulty_norm * error_map
    p0_high = component[:, high_band_slice(freq_bands)].sum(axis=1)
    p0_low_mid = component[:, low_mid_band_slice(freq_bands)].sum(axis=1)
    return p0_high, p0_low_mid, component


def _density_rows(runtime: CaseRuntime, args: argparse.Namespace, nfe: int, mode: str, strength: float, density: np.ndarray) -> list[dict[str, object]]:
    edges = logsnr_bin_edges(runtime.schedule, len(density), torch.device("cpu")).double()
    centers = 0.5 * (edges[:-1] + edges[1:])
    t_centers = runtime.schedule.inverse_log_snr(centers.float()).double()
    rows: list[dict[str, object]] = []
    for bin_index, value in enumerate(density):
        rows.append(
            {
                "case_id": runtime.spec.case_id,
                "dataset": runtime.spec.dataset,
                "model": runtime.spec.model_name,
                "solver": runtime.spec.solver,
                "nfe": nfe,
                "mode": mode,
                "strength": strength,
                "bin": bin_index,
                "logsnr_left": float(edges[bin_index].item()),
                "logsnr_right": float(edges[bin_index + 1].item()),
                "logsnr_center": float(centers[bin_index].item()),
                "t_center": float(t_centers[bin_index].item()),
                "density": float(value),
            }
        )
    return rows


def _gap_rows(runtime: CaseRuntime, nfe: int, mode: str, strength: float, times: torch.Tensor) -> list[dict[str, object]]:
    times_cpu = times.detach().float().cpu()
    lambdas = runtime.schedule.log_snr(times_cpu).detach().float().cpu()
    rows: list[dict[str, object]] = []
    for step in range(times_cpu.numel() - 1):
        rows.append(
            {
                "case_id": runtime.spec.case_id,
                "dataset": runtime.spec.dataset,
                "model": runtime.spec.model_name,
                "solver": runtime.spec.solver,
                "nfe": nfe,
                "mode": mode,
                "strength": strength,
                "step": step,
                "t_start": float(times_cpu[step].item()),
                "t_end": float(times_cpu[step + 1].item()),
                "t_gap": float((times_cpu[step] - times_cpu[step + 1]).item()),
                "logsnr_start": float(lambdas[step].item()),
                "logsnr_end": float(lambdas[step + 1].item()),
                "logsnr_gap": float((lambdas[step + 1] - lambdas[step]).item()),
            }
        )
    return rows


def _grid_sacrifice_row(
    runtime: CaseRuntime,
    args: argparse.Namespace,
    nfe: int,
    density: np.ndarray,
    p0_high: np.ndarray,
    p0_low_mid: np.ndarray,
    error_map: np.ndarray,
) -> dict[str, object]:
    excess = np.maximum(density - 1.0, 0.0)
    loss = np.maximum(1.0 - density, 0.0)
    high_norm = p0_high / max(float(p0_high.sum()), 1e-12)
    low_mid_norm = p0_low_mid / max(float(p0_low_mid.sum()), 1e-12)
    error_by_time = error_map.sum(axis=1)
    high_error_fraction = error_map[:, high_band_slice(args.freq_bands)].sum(axis=1) / np.maximum(error_by_time, 1e-12)
    low_mid_error_fraction = error_map[:, low_mid_band_slice(args.freq_bands)].sum(axis=1) / np.maximum(error_by_time, 1e-12)
    high_gain_score = float((excess * high_norm).sum())
    low_mid_sacrifice_score = float((loss * low_mid_norm).sum())
    return {
        "case_id": runtime.spec.case_id,
        "dataset": runtime.spec.dataset,
        "model": runtime.spec.model_name,
        "solver": runtime.spec.solver,
        "nfe": nfe,
        "strength": args.p0_high_strength,
        "density_argmax_bin": int(density.argmax()),
        "density_bin0": float(density[0]),
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "high_gain_score": high_gain_score,
        "low_mid_sacrifice_score": low_mid_sacrifice_score,
        "gain_minus_sacrifice": high_gain_score - low_mid_sacrifice_score,
        "gain_to_sacrifice_ratio": high_gain_score / max(low_mid_sacrifice_score, 1e-12),
        "mean_high_error_fraction": float(high_error_fraction.mean()),
        "max_high_error_fraction": float(high_error_fraction.max()),
        "mean_low_mid_error_fraction": float(low_mid_error_fraction.mean()),
        "p0_high_argmax_bin": int(p0_high.argmax()),
        "p0_high_sum": float(p0_high.sum()),
        "p0_low_mid_sum": float(p0_low_mid.sum()),
    }


def _collect_generation_inputs(runtime: CaseRuntime, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = []
    for _x, y in _limited_batches(runtime.loader, args.num_probe, torch.device("cpu")):
        labels.append(y.cpu())
    if not labels:
        raise RuntimeError(f"No labels available for spectral trace: {runtime.spec.case_id}")
    labels_all = torch.cat(labels, dim=0)[: args.num_probe].contiguous()
    stable_offset = sum((index + 1) * ord(char) for index, char in enumerate(runtime.spec.case_id)) % 10_000
    generator = torch.Generator(device="cpu").manual_seed(args.seed + stable_offset)
    noise = torch.randn(args.num_probe, 3, runtime.spec.image_size, runtime.spec.image_size, generator=generator)
    return noise, labels_all


@torch.no_grad()
def compute_spectral_trace(
    runtime: CaseRuntime,
    args: argparse.Namespace,
    nfe: int,
    mode: str,
    strength: float,
    density: np.ndarray,
    band_spec,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    intervals = _macro_intervals(runtime.spec.solver, nfe)
    times = _time_grid_from_density(runtime.schedule, intervals, density, device)
    noise_all, labels_all = _collect_generation_inputs(runtime, args, device)
    x0_hat_sums = torch.zeros(times.numel(), args.freq_bands, device=device)
    state_sums = torch.zeros(times.numel(), args.freq_bands, device=device)
    counts = torch.zeros(times.numel(), device=device)
    for start in range(0, args.num_probe, runtime.batch_size):
        end = min(start + runtime.batch_size, args.num_probe)
        x = noise_all[start:end].to(device, non_blocking=True)
        y = labels_all[start:end].to(device, non_blocking=True)
        for position in range(times.numel()):
            t = _batch_time(times[position], x.shape[0])
            eps = _predict_eps(runtime.model, x, t, y)
            x0_hat = runtime.schedule.eps_to_x0(x, t, eps)
            x0_hat_sums[position] += fft_radial_band_energy(x0_hat, band_spec).sum(dim=0)
            state_sums[position] += fft_radial_band_energy(x, band_spec).sum(dim=0)
            counts[position] += x.shape[0]
            if position + 1 < times.numel():
                full_corrector = True
                if runtime.spec.solver == "heun":
                    full_corrector = _heun_interval_is_full_corrector(nfe, position)
                x = _solver_step(
                    runtime.model,
                    runtime.schedule,
                    x,
                    t,
                    _batch_time(times[position + 1], x.shape[0]),
                    y,
                    runtime.spec.solver,
                    heun_full_corrector=full_corrector,
                )
    x0_hat_mean = (x0_hat_sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()
    state_mean = (state_sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()
    times_cpu = times.detach().float().cpu()
    lambdas = runtime.schedule.log_snr(times_cpu).detach().float().cpu()
    band_names = frequency_band_names(args.freq_bands)
    trace_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    for position in range(times.numel()):
        for band_index, band_name in enumerate(band_names):
            trace_rows.append(
                {
                    "case_id": runtime.spec.case_id,
                    "dataset": runtime.spec.dataset,
                    "model": runtime.spec.model_name,
                    "solver": runtime.spec.solver,
                    "nfe": nfe,
                    "mode": mode,
                    "strength": strength,
                    "position": position,
                    "t": float(times_cpu[position].item()),
                    "logsnr": float(lambdas[position].item()),
                    "band": band_index,
                    "band_name": band_name,
                    "x0_hat_energy": float(x0_hat_mean[position, band_index]),
                    "state_energy": float(state_mean[position, band_index]),
                }
            )
            if position > 0:
                transfer_rows.append(
                    {
                        "case_id": runtime.spec.case_id,
                        "dataset": runtime.spec.dataset,
                        "model": runtime.spec.model_name,
                        "solver": runtime.spec.solver,
                        "nfe": nfe,
                        "mode": mode,
                        "strength": strength,
                        "step": position - 1,
                        "band": band_index,
                        "band_name": band_name,
                        "delta_x0_hat_energy": float(x0_hat_mean[position, band_index] - x0_hat_mean[position - 1, band_index]),
                        "relative_delta_x0_hat_energy": float(
                            (x0_hat_mean[position, band_index] - x0_hat_mean[position - 1, band_index])
                            / max(abs(float(x0_hat_mean[position - 1, band_index])), 1e-12)
                        ),
                    }
                )
    return trace_rows, transfer_rows


def append_band_error_rows(
    rows: list[dict[str, object]],
    runtime: CaseRuntime,
    args: argparse.Namespace,
    nfe: int,
    difficulty: np.ndarray,
    maps: dict[str, np.ndarray],
    component: np.ndarray,
) -> None:
    band_names = frequency_band_names(args.freq_bands)
    difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
    error_map = maps["abs_error"]
    error_by_time = error_map.sum(axis=1)
    for bin_index in range(args.time_bins):
        for band_index, band_name in enumerate(band_names):
            rows.append(
                {
                    "case_id": runtime.spec.case_id,
                    "dataset": runtime.spec.dataset,
                    "model": runtime.spec.model_name,
                    "solver": runtime.spec.solver,
                    "nfe": nfe,
                    "time_bin": bin_index,
                    "band": band_index,
                    "band_name": band_name,
                    "difficulty": float(difficulty[bin_index, band_index]),
                    "difficulty_norm": float(difficulty_norm[bin_index, band_index]),
                    "local_error": float(maps["abs_error"][bin_index, band_index]),
                    "normalized_local_error": float(maps["norm_error"][bin_index, band_index]),
                    "ref_energy": float(maps["ref_energy"][bin_index, band_index]),
                    "start_energy": float(maps["start_energy"][bin_index, band_index]),
                    "difficulty_weighted_component": float(component[bin_index, band_index]),
                    "band_error_fraction_within_time": float(
                        maps["abs_error"][bin_index, band_index] / max(float(error_by_time[bin_index]), 1e-12)
                    ),
                    "macro_interval_index": int(maps["interval_indices"][bin_index]),
                    "heun_full_corrector": int(maps["heun_full_corrector"][bin_index]),
                    "num_probe": args.num_probe,
                    "ref_substeps": args.ref_substeps,
                    "time_bins": args.time_bins,
                    "freq_bands": args.freq_bands,
                    "checkpoint": runtime.checkpoint,
                }
            )


def append_p0_rows(
    rows: list[dict[str, object]],
    runtime: CaseRuntime,
    args: argparse.Namespace,
    nfe: int,
    p0_high: np.ndarray,
    p0_low_mid: np.ndarray,
    component: np.ndarray,
    error_map: np.ndarray,
) -> None:
    rows.append(
        {
            "case_id": runtime.spec.case_id,
            "dataset": runtime.spec.dataset,
            "model": runtime.spec.model_name,
            "solver": runtime.spec.solver,
            "nfe": nfe,
            "difficulty_weighted_error": float(component.sum()),
            "p0_high_sum": float(p0_high.sum()),
            "p0_low_mid_sum": float(p0_low_mid.sum()),
            "p0_high_fraction": float(p0_high.sum() / max(float(component.sum()), 1e-12)),
            "total_error": float(error_map.sum()),
            "argmax_bin": int(p0_high.argmax()),
            "num_probe": args.num_probe,
            "ref_substeps": args.ref_substeps,
            "time_bins": args.time_bins,
            "freq_bands": args.freq_bands,
            "checkpoint": runtime.checkpoint,
        }
    )


def plot_dataset_profiles(rows: list[dict[str, object]], summary_rows: list[dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        radius = np.asarray([float(row["radius_center"]) for row in selected])
        power = np.asarray([float(row["mean_power"]) for row in selected])
        ax.plot(radius[1:], power[1:], marker="o", linewidth=1.8, markersize=3.5, label=dataset)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("radial frequency")
    ax.set_ylabel("mean FFT power")
    ax.set_title("Dataset radial power spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "dataset_radial_power_spectrum.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    labels = [str(row["dataset"]) for row in summary_rows]
    values = [float(row["high_energy_ratio"]) for row in summary_rows]
    ax.bar(labels, values, color=["#2563eb", "#16a34a"][: len(values)])
    ax.set_ylabel("high-frequency energy ratio")
    ax.set_title("Dataset high-frequency energy")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(figure_dir / "dataset_high_frequency_ratio.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_case_heatmap(runtime: CaseRuntime, args: argparse.Namespace, nfe: int, difficulty: np.ndarray, error_map: np.ndarray, component: np.ndarray, output_dir: Path) -> None:
    figure_dir = output_dir / "figures" / "case_heatmaps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    matrices = [
        ("difficulty", difficulty),
        ("local_error", error_map),
        ("difficulty_weighted", component),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for ax, (name, matrix) in zip(axes, matrices):
        image = ax.imshow(np.log10(matrix + 1e-12), aspect="auto", origin="upper", cmap="magma")
        ax.set_title(name)
        ax.set_xlabel("frequency band")
        ax.set_ylabel("time bin")
        ax.set_xticks(range(args.freq_bands))
        ax.set_yticks(range(args.time_bins))
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.suptitle(f"{runtime.spec.case_id} NFE={nfe}")
    fig.savefig(figure_dir / f"{runtime.spec.case_id}_nfe{nfe}_maps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_trace(trace_rows: list[dict[str, object]], runtime: CaseRuntime, nfe: int, output_dir: Path) -> None:
    figure_dir = output_dir / "figures" / "spectral_transfer"
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in trace_rows if row["case_id"] == runtime.spec.case_id and int(row["nfe"]) == nfe]
    if not selected:
        return
    modes = sorted({(str(row["mode"]), float(row["strength"])) for row in selected}, key=lambda item: (item[0], item[1]))
    bands = sorted({int(row["band"]) for row in selected})
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (mode, strength) in zip(axes, modes):
        for band in bands:
            points = sorted(
                [
                    row
                    for row in selected
                    if str(row["mode"]) == mode and float(row["strength"]) == strength and int(row["band"]) == band
                ],
                key=lambda row: int(row["position"]),
            )
            ax.plot(
                [int(row["position"]) for row in points],
                [float(row["x0_hat_energy"]) for row in points],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=str(points[0]["band_name"]) if points else f"band {band}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("macro step position")
        ax.set_title(f"{mode} s={strength:g}")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("x0_hat band energy")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{runtime.spec.case_id} spectral trace NFE={nfe}")
    fig.savefig(figure_dir / f"{runtime.spec.case_id}_nfe{nfe}_spectral_trace.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_grid_sacrifice(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for row in rows:
        ax.scatter(float(row["low_mid_sacrifice_score"]), float(row["high_gain_score"]), s=64)
        ax.text(
            float(row["low_mid_sacrifice_score"]),
            float(row["high_gain_score"]),
            f"{row['model']} {row['solver']}@{row['nfe']}",
            fontsize=7,
            alpha=0.82,
        )
    ax.set_xlabel("low/mid sacrifice score")
    ax.set_ylabel("high-frequency gain score")
    ax.set_title("p0_high grid sacrifice diagnostic")
    ax.grid(True, alpha=0.25)
    fig.savefig(figure_dir / "grid_sacrifice_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_case(runtime: CaseRuntime, args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    band_spec = radial_band_spec(runtime.spec.image_size, runtime.spec.image_size, args.freq_bands, device)
    print(f"case_start case={runtime.spec.case_id} checkpoint={runtime.checkpoint}", flush=True)
    difficulty = compute_difficulty_map(runtime, args, band_spec, device)
    maps_payload: dict[str, np.ndarray] = {"difficulty": difficulty}
    band_rows: list[dict[str, object]] = []
    p0_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    sacrifice_rows: list[dict[str, object]] = []
    trace_rows_all: list[dict[str, object]] = []
    transfer_rows_all: list[dict[str, object]] = []

    for nfe in runtime.spec.nfes:
        print(f"band_error_start case={runtime.spec.case_id} nfe={nfe}", flush=True)
        maps = compute_bandwise_local_error(runtime, args, nfe, band_spec, device)
        p0_high, p0_low_mid, component = _profile_from_maps(difficulty, maps["abs_error"], args.freq_bands)
        append_band_error_rows(band_rows, runtime, args, nfe, difficulty, maps, component)
        append_p0_rows(p0_rows, runtime, args, nfe, p0_high, p0_low_mid, component, maps["abs_error"])
        maps_payload[f"abs_error_nfe_{nfe}"] = maps["abs_error"]
        maps_payload[f"norm_error_nfe_{nfe}"] = maps["norm_error"]
        maps_payload[f"component_nfe_{nfe}"] = component
        maps_payload[f"p0_high_nfe_{nfe}"] = p0_high
        plot_case_heatmap(runtime, args, nfe, difficulty, maps["abs_error"], component, output_dir)

        uniform_density = np.ones(args.time_bins, dtype=np.float64)
        p0_density = _normalize_profile(p0_high, args.p0_high_strength, args.max_density_ratio)
        for mode, strength, density in [
            ("uniform", 0.0, uniform_density),
            ("p0_high", args.p0_high_strength, p0_density),
        ]:
            intervals = _macro_intervals(runtime.spec.solver, nfe)
            times = _time_grid_from_density(runtime.schedule, intervals, density, device)
            density_rows.extend(_density_rows(runtime, args, nfe, mode, strength, density))
            gap_rows.extend(_gap_rows(runtime, nfe, mode, strength, times))
            print(f"spectral_transfer_start case={runtime.spec.case_id} nfe={nfe} mode={mode}", flush=True)
            trace_rows, transfer_rows = compute_spectral_trace(runtime, args, nfe, mode, strength, density, band_spec, device)
            trace_rows_all.extend(trace_rows)
            transfer_rows_all.extend(transfer_rows)
        sacrifice_rows.append(_grid_sacrifice_row(runtime, args, nfe, p0_density, p0_high, p0_low_mid, maps["abs_error"]))
        plot_spectral_trace(trace_rows_all, runtime, nfe, output_dir)

        case_dir = ensure_dir(output_dir / "cases" / runtime.spec.case_id)
        _write_csv(case_dir / "bandwise_local_error.csv", band_rows)
        _write_csv(case_dir / "p0_frequency_scores.csv", p0_rows)
        _write_csv(case_dir / "timegrid_density.csv", density_rows)
        _write_csv(case_dir / "timegrid_gaps.csv", gap_rows)
        _write_csv(case_dir / "grid_sacrifice.csv", sacrifice_rows)
        _write_csv(case_dir / "spectral_trace.csv", trace_rows_all)
        _write_csv(case_dir / "spectral_transfer.csv", transfer_rows_all)
        np.savez_compressed(case_dir / "frequency_maps.npz", **maps_payload)
        print(
            f"case_progress case={runtime.spec.case_id} nfe={nfe} "
            f"p0_high_sum={float(p0_high.sum()):.6e} density_argmax={int(p0_density.argmax())}",
            flush=True,
        )
    plot_grid_sacrifice(sacrifice_rows, output_dir / "cases" / runtime.spec.case_id)
    print(f"case_done case={runtime.spec.case_id}", flush=True)


def merge_case_csvs(output_dir: Path) -> None:
    targets = [
        "bandwise_local_error.csv",
        "p0_frequency_scores.csv",
        "timegrid_density.csv",
        "timegrid_gaps.csv",
        "grid_sacrifice.csv",
        "spectral_trace.csv",
        "spectral_transfer.csv",
    ]
    for target in targets:
        rows: list[dict[str, object]] = []
        for path in sorted((output_dir / "cases").glob(f"*/{target}")):
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        if rows:
            _write_csv(output_dir / target, rows)
    if (output_dir / "grid_sacrifice.csv").exists():
        with (output_dir / "grid_sacrifice.csv").open("r", encoding="utf-8", newline="") as handle:
            plot_grid_sacrifice(list(csv.DictReader(handle)), output_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    print(f"frequency_diagnosis_start device={device} output_dir={output_dir}", flush=True)
    print(
        f"settings num_probe={args.num_probe} ref_substeps={args.ref_substeps} "
        f"time_bins={args.time_bins} freq_bands={args.freq_bands} strength={args.p0_high_strength}",
        flush=True,
    )
    if "dataset_profile" in args.stages:
        compute_dataset_profiles(args, output_dir, device)

    for case_id in args.cases:
        spec = CASE_SPECS[case_id]
        try:
            runtime = _load_case_runtime(args, spec, device)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"case_skip_missing case={case_id} error={exc}", flush=True)
                continue
            raise
        run_case(runtime, args, output_dir, device)
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    merge_case_csvs(output_dir)
    print(f"frequency_diagnosis_done output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
