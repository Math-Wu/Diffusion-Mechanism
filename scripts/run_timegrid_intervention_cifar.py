from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import logsnr_bin_edges
from dm.eval_utils import (
    default_noise_bank_path,
    default_real_stats_path,
    load_or_compute_real_stats,
    load_or_create_noise_bank,
    noise_bank_id,
    noise_batches,
)
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.metrics import build_feature_extractor, feature_stats, frechet_distance
from dm.samplers.base import CountedModel, SamplerResult
from dm.samplers.ode import AB_COEFFS
from dm.utils import default_device, ensure_dir, set_seed


SUPPORTED_SOLVERS = ("ddim", "heun", "dpmpp", "unipc")
INTERVENTION_MODES = ("uniform", "p0_all", "p0_high", "p2_x0_high", "joint_high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mechanism-guided nonuniform time-grid interventions on CIFAR.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/timegrid_intervention_cifar_medium")
    parser.add_argument("--p0_maps", default="outputs/mechanism_p0_cifar_medium_fixedbin/mechanism_maps.npz")
    parser.add_argument("--p2_maps", default="outputs/trajectory_mechanisms_cifar_medium/trajectory_maps.npz")
    parser.add_argument("--modes", nargs="+", default=["uniform", "p0_high", "p2_x0_high", "joint_high"], choices=INTERVENTION_MODES)
    parser.add_argument("--solvers", nargs="+", default=["dpmpp", "unipc"], choices=SUPPORTED_SOLVERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[8, 15, 20])
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint_name", default="last.pt")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    parser.add_argument("--noise_bank")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--profile_strength", type=float, default=0.6)
    parser.add_argument("--max_density_ratio", type=float, default=4.0)
    parser.add_argument("--smooth_profile", action="store_true")
    parser.add_argument("--raw_weights", action="store_true")
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _macro_intervals(solver: str, nfe: int) -> int:
    return max(1, (nfe + 1) // 2) if solver == "heun" else max(1, nfe)


def high_band_slice(num_bands: int) -> slice:
    start = max(0, int(np.ceil(num_bands * 0.625)))
    return slice(start, num_bands)


def normalize_profile(profile: np.ndarray, *, strength: float, max_density_ratio: float, smooth: bool) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    profile = np.clip(profile, a_min=0.0, a_max=None)
    if smooth and profile.size >= 3:
        padded = np.pad(profile, (1, 1), mode="edge")
        profile = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    if float(profile.sum()) <= 1e-12:
        profile = np.ones_like(profile)
    profile = profile / max(float(profile.mean()), 1e-12)
    profile = np.clip(profile, 1.0 / max_density_ratio, max_density_ratio)
    density = (1.0 - strength) + strength * profile
    return density / max(float(density.mean()), 1e-12)


def map_indices(maps: np.lib.npyio.NpzFile) -> tuple[dict[str, int], dict[str, int], dict[int, int]]:
    arch_to_idx = {str(item): index for index, item in enumerate(maps["architectures"])}
    solver_to_idx = {str(item): index for index, item in enumerate(maps["solvers"])}
    nfe_to_idx = {int(item): index for index, item in enumerate(maps["nfes"])}
    return arch_to_idx, solver_to_idx, nfe_to_idx


def profile_from_maps(
    p0_maps: np.lib.npyio.NpzFile,
    p2_maps: np.lib.npyio.NpzFile,
    architecture: str,
    solver: str,
    nfe: int,
    mode: str,
    *,
    strength: float,
    max_density_ratio: float,
    smooth: bool,
) -> np.ndarray:
    bins = int(p0_maps["difficulty"].shape[1])
    if mode == "uniform":
        return np.ones(bins, dtype=np.float64)
    p0_arch, p0_solver, p0_nfe = map_indices(p0_maps)
    p2_arch, p2_solver, p2_nfe = map_indices(p2_maps)
    ai = p0_arch[architecture]
    si = p0_solver[solver]
    ni = p0_nfe[nfe]
    difficulty = p0_maps["difficulty"][ai]
    difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
    solver_error = p0_maps["solver_error"][ai, si, ni]
    p0_component = difficulty_norm * solver_error
    high_slice = high_band_slice(p0_component.shape[1])

    if mode == "p0_all":
        profile = p0_component.sum(axis=1)
    elif mode == "p0_high":
        profile = p0_component[:, high_slice].sum(axis=1)
    else:
        p2_ai = p2_arch[architecture]
        p2_si = p2_solver[solver]
        p2_ni = p2_nfe[nfe]
        p2_high = p2_maps["x0_freq_error"][p2_ai, p2_si, p2_ni, :, high_slice].sum(axis=1)
        if mode == "p2_x0_high":
            profile = p2_high
        elif mode == "joint_high":
            p0_high = p0_component[:, high_slice].sum(axis=1)
            p0_norm = p0_high / max(float(p0_high.mean()), 1e-12)
            p2_norm = p2_high / max(float(p2_high.mean()), 1e-12)
            profile = 0.5 * p0_norm + 0.5 * p2_norm
        else:
            raise ValueError(f"Unknown intervention mode: {mode}")
    return normalize_profile(profile, strength=strength, max_density_ratio=max_density_ratio, smooth=smooth)


def time_grid_from_density(schedule, intervals: int, density: np.ndarray, device: torch.device) -> torch.Tensor:
    if intervals < 1:
        raise ValueError("intervals must be >= 1")
    edges_t = logsnr_bin_edges(schedule, len(density), device=torch.device("cpu")).double().numpy()
    widths = np.diff(edges_t)
    mass = np.maximum(density, 1e-8) * widths
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    targets = np.linspace(0.0, float(cdf[-1]), intervals + 1)
    lambdas = np.interp(targets, cdf, edges_t)
    lambdas_t = torch.tensor(lambdas, device=device, dtype=torch.float32)
    return schedule.inverse_log_snr(lambdas_t)


@torch.no_grad()
def sample_custom_grid(model, schedule, x: torch.Tensor, solver: str, nfe: int, times: torch.Tensor) -> SamplerResult:
    counted = CountedModel(model)
    batch = x.shape[0]
    if solver == "ddim":
        for index in range(nfe):
            t = _batch_time(times[index], batch)
            t_next = _batch_time(times[index + 1], batch)
            eps = counted(x, t)
            x0 = schedule.eps_to_x0(x, t, eps)
            alpha_next, sigma_next = schedule.alpha_sigma(t_next)
            while alpha_next.ndim < x.ndim:
                alpha_next = alpha_next[..., None]
                sigma_next = sigma_next[..., None]
            x = alpha_next * x0 + sigma_next * eps
        return SamplerResult(samples=x, nfe=counted.nfe)

    if solver == "heun":
        remaining = nfe
        intervals = _macro_intervals(solver, nfe)
        for index in range(intervals):
            t = _batch_time(times[index], batch)
            t_next = _batch_time(times[index + 1], batch)
            dt = (t_next - t).view(batch, *([1] * (x.ndim - 1)))
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

    max_order = 2 if solver == "dpmpp" else 3
    history: list[torch.Tensor] = []
    for index in range(nfe):
        t = _batch_time(times[index], batch)
        t_next = _batch_time(times[index + 1], batch)
        dt = (t_next - t).view(batch, *([1] * (x.ndim - 1)))
        eps = counted(x, t)
        drift = schedule.drift(x, t, eps)
        history.insert(0, drift)
        order = min(max_order, len(history))
        update = torch.zeros_like(x)
        for coeff, old_drift in zip(AB_COEFFS[order], history):
            update = update + coeff * old_drift
        x = x + dt * update
        history = history[:max_order]
    return SamplerResult(samples=x, nfe=counted.nfe)


@torch.no_grad()
def generated_feature_stats(model, schedule, extractor, solver, nfe, times, batch_size, noise_bank, total, device):
    features = []
    runtime = 0.0
    total_nfe = 0
    for noise in noise_batches(noise_bank, batch_size, device):
        start = time.time()
        result = sample_custom_grid(model, schedule, noise, solver=solver, nfe=nfe, times=times)
        runtime += time.time() - start
        total_nfe += result.nfe
        features.append(extractor(result.samples).detach().float().cpu().numpy())
    return feature_stats(np.concatenate(features, axis=0)[:total]), runtime, total_nfe


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.run_dirs):
        raise ValueError("--configs and --run_dirs must have the same length")
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    p0_maps = np.load(args.p0_maps)
    p2_maps = np.load(args.p2_maps)
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    rows: list[dict[str, str | int | float]] = []
    real_stats_cache = None
    real_stats_path = None
    noise_bank = None
    noise_path = None
    noise_id = None

    for config_path, run_dir in zip(args.configs, args.run_dirs):
        checkpoint = checkpoint_path_for_run(run_dir, name=args.checkpoint_name)
        model, config, ckpt_payload = load_model_from_checkpoint(config_path, checkpoint, device, use_ema=not args.raw_weights)
        config["training"]["batch_size"] = args.batch_size
        schedule = build_schedule(config)
        architecture = str(config["model"]["architecture"])
        image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
        noise_shape = (int(config["model"]["in_channels"]), image_size, image_size)
        _, val_loader = __import__("dm.data", fromlist=["build_cifar10_loaders"]).build_cifar10_loaders(config, download=False)
        if real_stats_cache is None:
            real_stats_path = args.real_stats_cache or default_real_stats_path(
                config["data"].get("root", "data"),
                split_seed=int(config["data"].get("split_seed", 20260423)),
                num_samples=args.num_samples,
                feature_backend=args.feature_backend,
            )
            real_stats_cache = load_or_compute_real_stats(
                real_stats_path,
                itertools.cycle(val_loader),
                extractor,
                device,
                args.num_samples,
                metadata={
                    "dataset": "cifar10",
                    "split_seed": int(config["data"].get("split_seed", 20260423)),
                    "feature_backend": args.feature_backend,
                    "num_samples": args.num_samples,
                },
            )
        if noise_bank is None:
            noise_path = args.noise_bank or default_noise_bank_path(
                config["data"].get("root", "data"), args.seed, args.num_samples, noise_shape
            )
            noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=noise_shape, seed=args.seed)
            noise_id = noise_bank_id(noise_path)
        checkpoint_id = f"step_{int(ckpt_payload.get('step', -1))}_images_{int(ckpt_payload.get('images_seen', -1))}"

        for solver in args.solvers:
            for nfe in args.nfe:
                intervals = _macro_intervals(solver, nfe)
                baseline_fid = None
                for mode in args.modes:
                    density = profile_from_maps(
                        p0_maps,
                        p2_maps,
                        architecture,
                        solver,
                        nfe,
                        mode,
                        strength=args.profile_strength,
                        max_density_ratio=args.max_density_ratio,
                        smooth=args.smooth_profile,
                    )
                    times = time_grid_from_density(schedule, intervals, density, device)
                    stats, runtime, total_nfe = generated_feature_stats(
                        model,
                        schedule,
                        extractor,
                        solver,
                        nfe,
                        times,
                        args.batch_size,
                        noise_bank,
                        args.num_samples,
                        device,
                    )
                    fid = frechet_distance(real_stats_cache, stats)
                    if mode == "uniform":
                        baseline_fid = fid
                    rows.append(
                        {
                            "architecture": architecture,
                            "solver": solver,
                            "nfe": nfe,
                            "intervention_mode": mode,
                            "fid": fid,
                            "delta_vs_uniform": fid - baseline_fid if baseline_fid is not None else 0.0,
                            "wall_clock_sec": runtime,
                            "total_model_calls": total_nfe,
                            "num_samples": args.num_samples,
                            "profile_strength": args.profile_strength,
                            "max_density_ratio": args.max_density_ratio,
                            "smooth_profile": int(args.smooth_profile),
                            "density_min": float(density.min()),
                            "density_max": float(density.max()),
                            "density_argmax_bin": int(density.argmax()),
                            "checkpoint": str(checkpoint),
                            "checkpoint_id": checkpoint_id,
                            "seed": args.seed,
                            "noise_bank_id": noise_id or "",
                            "noise_bank": str(noise_path),
                            "real_stats_cache": str(real_stats_path),
                            "feature_backend": args.feature_backend,
                        }
                    )
                    print(
                        f"{architecture} {solver} nfe={nfe} mode={mode} fid={fid:.4f} "
                        f"delta_vs_uniform={rows[-1]['delta_vs_uniform']:.4f}",
                        flush=True,
                    )

    output = output_dir / "timegrid_intervention_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output, flush=True)


if __name__ == "__main__":
    main()
