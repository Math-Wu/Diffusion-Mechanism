from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dm_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import fft_radial_band_energy, radial_band_spec
from dm.eval_utils import load_or_create_noise_bank, noise_bank_id
from dm.imagenet256 import decode_latents
from dm.imagenet256_flow import MODEL_SHAPES, load_autoencoder_kl, load_sit_xl_2
from dm.metrics import FeatureStats, build_feature_extractor, feature_stats, frechet_distance
from dm.utils import default_device, ensure_dir, set_seed

from run_imagenet256_pretrained_sweep import _default_noise_bank_path, _real_image_batches_from_npz, _save_grid


SAMPLERS = ("euler", "heun", "rk2")
GRIDS = ("uniform", "official", "p0_high", "curvature_high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet256 SiT/flow low-NFE validation with mechanism-guided grids.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samplers", nargs="+", choices=SAMPLERS, default=list(SAMPLERS))
    parser.add_argument("--grids", nargs="+", choices=GRIDS, default=list(GRIDS))
    parser.add_argument("--nfe", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_real_stats", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--probe_noise_bank")
    parser.add_argument("--num_probe", type=int, default=128)
    parser.add_argument("--probe_batch_size", type=int, default=None)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--profile_strength", type=float, default=0.5)
    parser.add_argument("--max_density_ratio", type=float, default=2.5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--latent_scale_factor", type=float, default=0.18215)
    parser.add_argument("--real_npz", default="data/imagenet256/ref_batches/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--real_features_cache")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    parser.add_argument("--metric_subset", type=int, default=2048)
    parser.add_argument("--density_k", type=int, default=5)
    parser.add_argument("--fdd_bands", type=int, default=16)
    parser.add_argument("--grid_samples", type=int, default=64)
    parser.add_argument("--grid_nrow", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare_real_features_only", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _intervals_for_nfe(sampler: str, nfe: int) -> int:
    if sampler == "euler":
        return max(1, nfe)
    return max(1, nfe // 2)


def _effective_nfe(sampler: str, intervals: int) -> int:
    return intervals if sampler == "euler" else 2 * intervals


def _normalize_profile(profile: np.ndarray, strength: float, max_density_ratio: float) -> np.ndarray:
    profile = np.clip(np.asarray(profile, dtype=np.float64), 0.0, None)
    if profile.size >= 3:
        padded = np.pad(profile, (1, 1), mode="edge")
        profile = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    if float(profile.sum()) <= 1e-12:
        profile = np.ones_like(profile)
    profile = profile / max(float(profile.mean()), 1e-12)
    profile = np.clip(profile, 1.0 / max_density_ratio, max_density_ratio)
    density = (1.0 - strength) + strength * profile
    return density / max(float(density.mean()), 1e-12)


def _time_grid_from_density(intervals: int, density: np.ndarray, device: torch.device) -> torch.Tensor:
    edges = np.linspace(0.0, 1.0, len(density) + 1, dtype=np.float64)
    widths = np.diff(edges)
    mass = np.maximum(density, 1e-8) * widths
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    targets = np.linspace(0.0, float(cdf[-1]), intervals + 1)
    return torch.tensor(np.interp(targets, cdf, edges), device=device, dtype=torch.float32)


def _uniform_time_grid(intervals: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, intervals + 1, device=device)


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


@torch.no_grad()
def _velocity(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled and device.type == "cuda"):
        if abs(cfg_scale - 1.0) <= 1e-12:
            return model(x, t, y).float()
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        y_null = torch.full_like(y, 1000)
        y_cat = torch.cat([y, y_null], dim=0)
        out = model.forward_with_cfg(x_cat, t_cat, y_cat, cfg_scale).float()
        return out[: x.shape[0]]


@torch.no_grad()
def _flow_step(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    sampler: str,
    cfg_scale: float,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    v1 = _velocity(model, x, t, y, cfg_scale, amp_enabled, device)
    if sampler == "euler":
        return x + dt * v1
    if sampler == "heun":
        x_euler = x + dt * v1
        v2 = _velocity(model, x_euler, t_next, y, cfg_scale, amp_enabled, device)
        return x + 0.5 * dt * (v1 + v2)
    if sampler == "rk2":
        t_mid = 0.5 * (t + t_next)
        x_mid = x + 0.5 * dt * v1
        v_mid = _velocity(model, x_mid, t_mid, y, cfg_scale, amp_enabled, device)
        return x + dt * v_mid
    raise ValueError(f"Unknown sampler: {sampler}")


@torch.no_grad()
def _reference_heun(
    model,
    x: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    y: torch.Tensor,
    substeps: int,
    cfg_scale: float,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    times = torch.linspace(float(t_start[0]), float(t_end[0]), substeps + 1, device=x.device)
    current = x
    for index in range(substeps):
        current = _flow_step(
            model,
            current,
            _batch_time(times[index], current.shape[0]),
            _batch_time(times[index + 1], current.shape[0]),
            y,
            "heun",
            cfg_scale,
            amp_enabled,
            device,
        )
    return current


@torch.no_grad()
def _sample_latents(
    model,
    noise: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    sampler: str,
    cfg_scale: float,
    amp_enabled: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    x = noise
    calls = 0
    for index in range(times.numel() - 1):
        t = _batch_time(times[index], x.shape[0])
        t_next = _batch_time(times[index + 1], x.shape[0])
        x = _flow_step(model, x, t, t_next, labels, sampler, cfg_scale, amp_enabled, device)
        calls += 1 if sampler == "euler" else 2
    return x, calls


@torch.no_grad()
def _decode(vae, latents: torch.Tensor, scale_factor: float, amp_enabled: bool, device: torch.device) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled and device.type == "cuda"):
        return decode_latents(vae, latents, scale_factor=scale_factor).float()


def _spectral_stats_update(images: torch.Tensor, band_spec, sums: torch.Tensor, sq_sums: torch.Tensor, count: int):
    power = fft_radial_band_energy(images, band_spec)
    sums += power.sum(dim=0)
    sq_sums += power.square().sum(dim=0)
    return count + images.shape[0]


def _spectral_stats_finalize(sums: torch.Tensor, sq_sums: torch.Tensor, count: int) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, count)
    mean = sums / count
    var = (sq_sums / count - mean.square()).clamp_min(0.0)
    return mean.detach().cpu().numpy(), var.sqrt().detach().cpu().numpy()


def _fdd(real_mean: np.ndarray, real_std: np.ndarray, fake_mean: np.ndarray, fake_std: np.ndarray) -> float:
    return float(np.mean((np.log1p(fake_mean) - np.log1p(real_mean)) ** 2 + (np.log1p(fake_std) - np.log1p(real_std)) ** 2))


@torch.no_grad()
def _load_or_create_real_features_and_fdd(
    *,
    args: argparse.Namespace,
    extractor: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    fdd_band_spec,
) -> tuple[np.ndarray, FeatureStats, np.ndarray, np.ndarray, Path]:
    cache = Path(args.real_features_cache) if args.real_features_cache else output_dir / f"imagenet256_real_{args.feature_backend}_features_n{args.num_real_stats}_fdd{args.fdd_bands}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        features = data["features"]
        return features, feature_stats(features), data["fdd_mean"], data["fdd_std"], cache

    features: list[np.ndarray] = []
    fdd_sums = torch.zeros(args.fdd_bands, device=device)
    fdd_sq_sums = torch.zeros(args.fdd_bands, device=device)
    fdd_count = 0
    seen = 0
    for batch in _real_image_batches_from_npz(args.real_npz, args.num_real_stats, args.real_batch_size):
        batch = batch.to(device, non_blocking=True)
        features.append(extractor(batch).detach().float().cpu().numpy())
        fdd_count = _spectral_stats_update(batch, fdd_band_spec, fdd_sums, fdd_sq_sums, fdd_count)
        seen += batch.shape[0]
        print(f"real_feature_progress {seen}/{args.num_real_stats}", flush=True)
    feats = np.concatenate(features, axis=0)[: args.num_real_stats]
    fdd_mean, fdd_std = _spectral_stats_finalize(fdd_sums, fdd_sq_sums, fdd_count)
    ensure_dir(cache.parent)
    np.savez_compressed(
        cache,
        features=feats,
        fdd_mean=fdd_mean,
        fdd_std=fdd_std,
        feature_backend=args.feature_backend,
        num_real_stats=args.num_real_stats,
        real_npz=str(args.real_npz),
    )
    return feats, feature_stats(feats), fdd_mean, fdd_std, cache


def _subsample_features(real_features: np.ndarray, fake_features: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(max_samples, len(real_features), len(fake_features))
    real_idx = rng.choice(len(real_features), size=n, replace=False)
    fake_idx = rng.choice(len(fake_features), size=n, replace=False)
    return real_features[real_idx].astype(np.float32), fake_features[fake_idx].astype(np.float32)


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray, chunk: int = 256) -> np.ndarray:
    x_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    blocks = []
    for start in range(0, x_t.shape[0], chunk):
        blocks.append(torch.cdist(x_t[start : start + chunk], y_t).square().numpy())
    return np.concatenate(blocks, axis=0)


def _cmmd_rbf(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    xx = _pairwise_sq_dists(real_features, real_features)
    yy = _pairwise_sq_dists(fake_features, fake_features)
    xy = _pairwise_sq_dists(real_features, fake_features)
    distances = np.concatenate([xx.reshape(-1), yy.reshape(-1), xy.reshape(-1)])
    sigma2 = float(np.median(distances[distances > 0]))
    sigma2 = max(sigma2, 1e-6)
    return float(np.exp(-xx / (2 * sigma2)).mean() + np.exp(-yy / (2 * sigma2)).mean() - 2.0 * np.exp(-xy / (2 * sigma2)).mean())


def _density_coverage(real_features: np.ndarray, fake_features: np.ndarray, k: int) -> tuple[float, float]:
    rr = _pairwise_sq_dists(real_features, real_features)
    rr.sort(axis=1)
    radii = rr[:, min(k, rr.shape[1] - 1)]
    rf = _pairwise_sq_dists(real_features, fake_features)
    density = float(((rf <= radii[:, None]).sum(axis=0) / max(k, 1)).mean())
    coverage = float((rf.min(axis=1) <= radii).mean())
    return density, coverage


@torch.no_grad()
def _profile_maps(
    model,
    probe_noise: torch.Tensor,
    labels: torch.Tensor,
    sampler: str,
    args: argparse.Namespace,
    band_spec,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, np.ndarray]:
    bins = args.time_bins
    freq = args.freq_bands
    edges = torch.linspace(0.0, 1.0, bins + 1, device=device)
    local_error = torch.zeros(bins, freq, device=device)
    curvature = torch.zeros(bins, freq, device=device)
    velocity_rms = torch.zeros(bins, device=device)
    counts = torch.zeros(bins, device=device)
    batch_size = args.probe_batch_size or args.batch_size
    for start in range(0, probe_noise.shape[0], batch_size):
        end = min(start + batch_size, probe_noise.shape[0])
        x = probe_noise[start:end].to(device, non_blocking=True)
        y = labels[start:end].to(device, non_blocking=True)
        for bin_index in range(bins):
            t = _batch_time(edges[bin_index], x.shape[0])
            t_next = _batch_time(edges[bin_index + 1], x.shape[0])
            x_ref_next = _reference_heun(model, x, t, t_next, y, args.ref_substeps, args.cfg_scale, amp_enabled, device)
            x_solver_next = _flow_step(model, x, t, t_next, y, sampler, args.cfg_scale, amp_enabled, device)
            v_start = _velocity(model, x, t, y, args.cfg_scale, amp_enabled, device)
            v_end = _velocity(model, x_ref_next, t_next, y, args.cfg_scale, amp_enabled, device)
            local_error[bin_index] += fft_radial_band_energy(x_solver_next - x_ref_next, band_spec).sum(dim=0)
            curvature[bin_index] += fft_radial_band_energy(v_end - v_start, band_spec).sum(dim=0)
            velocity_rms[bin_index] += v_start.flatten(1).square().mean(dim=1).sqrt().sum()
            counts[bin_index] += x.shape[0]
            x = x_ref_next
    denom = counts.clamp_min(1.0)
    return {
        "local_error": (local_error / denom[:, None]).detach().cpu().numpy(),
        "curvature": (curvature / denom[:, None]).detach().cpu().numpy(),
        "velocity_rms": (velocity_rms / denom).detach().cpu().numpy(),
    }


@torch.no_grad()
def _trajectory_drift(
    model,
    probe_noise: torch.Tensor,
    labels: torch.Tensor,
    sampler: str,
    times: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    total = 0.0
    count = 0
    batch_size = args.probe_batch_size or args.batch_size
    for start in range(0, probe_noise.shape[0], batch_size):
        end = min(start + batch_size, probe_noise.shape[0])
        x_solver = probe_noise[start:end].to(device, non_blocking=True)
        x_ref = x_solver.clone()
        y = labels[start:end].to(device, non_blocking=True)
        for index in range(times.numel() - 1):
            t = _batch_time(times[index], x_solver.shape[0])
            t_next = _batch_time(times[index + 1], x_solver.shape[0])
            x_solver = _flow_step(model, x_solver, t, t_next, y, sampler, args.cfg_scale, amp_enabled, device)
            x_ref = _reference_heun(model, x_ref, t, t_next, y, args.ref_substeps, args.cfg_scale, amp_enabled, device)
        total += float((x_solver - x_ref).flatten(1).square().mean(dim=1).sqrt().sum().detach().cpu())
        count += x_solver.shape[0]
    return total / max(count, 1)


def _high_slice(freq_bands: int) -> slice:
    return slice(max(0, int(math.ceil(freq_bands * 0.625))), freq_bands)


def _density_for_grid(grid: str, profile: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if grid in {"uniform", "official"}:
        return np.ones(args.time_bins, dtype=np.float64), "official_sit_uniform_time" if grid == "official" else "uniform_time"
    high = _high_slice(args.freq_bands)
    if grid == "p0_high":
        p = np.asarray(profile["local_error"], dtype=np.float64)[:, high].sum(axis=1)
        return _normalize_profile(p, args.profile_strength, args.max_density_ratio), "latent_local_high_frequency_error"
    if grid == "curvature_high":
        p = np.asarray(profile["curvature"], dtype=np.float64)[:, high].sum(axis=1)
        return _normalize_profile(p, args.profile_strength, args.max_density_ratio), "latent_velocity_curvature_high_frequency"
    raise ValueError(f"Unknown grid: {grid}")


def _density_rows(grid: str, sampler: str, nfe: int, density: np.ndarray, source: str) -> list[dict[str, object]]:
    return [
        {
            "model": "sit_xl_2",
            "sampler": sampler,
            "grid": grid,
            "nfe": nfe,
            "time_bin": index,
            "density": float(value),
            "density_source": source,
        }
        for index, value in enumerate(density)
    ]


def _correlation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    predictors = [
        "trajectory_drift",
        "density_weighted_local_high_error",
        "density_weighted_curvature_high",
        "density_weighted_d_flow",
        "fdd",
        "cmmd",
        "density",
        "coverage",
    ]
    out = []
    if not rows:
        return out
    groups: dict[str, list[dict[str, object]]] = {"all": rows}
    for sampler in sorted({str(row["sampler"]) for row in rows}):
        groups[f"sampler:{sampler}"] = [row for row in rows if row["sampler"] == sampler]
    for grid in sorted({str(row["grid"]) for row in rows}):
        groups[f"grid:{grid}"] = [row for row in rows if row["grid"] == grid]
    for group, group_rows in groups.items():
        y = np.asarray([float(row["delta_fid_to_best"]) for row in group_rows], dtype=np.float64)
        for predictor in predictors:
            x = np.asarray([float(row.get(predictor, float("nan"))) for row in group_rows], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3 or x[mask].std() <= 1e-12 or y[mask].std() <= 1e-12:
                pearson = float("nan")
            else:
                pearson = float(np.corrcoef(x[mask], y[mask])[0, 1])
            out.append(
                {
                    "group": group,
                    "predictor": predictor,
                    "target": "delta_fid_to_best",
                    "n": int(mask.sum()),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if math.isfinite(pearson) else float("nan"),
                }
            )
    return out


def _update_delta_to_best(rows: list[dict[str, object]]) -> None:
    finite_fids = [float(row["fid"]) for row in rows if math.isfinite(float(row["fid"]))]
    if not finite_fids:
        return
    best = min(finite_fids)
    for row in rows:
        row["delta_fid_to_best"] = float(row["fid"]) - best
        row["best_observed_fid"] = best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    amp_enabled = bool(args.amp and device.type == "cuda")
    output_dir = ensure_dir(args.output_dir)
    grid_dir = ensure_dir(output_dir / "sample_grids")
    print(
        f"flow_validation_start model=sit_xl_2 device={device} cfg={args.cfg_scale} "
        f"num_samples={args.num_samples} output_dir={output_dir}",
        flush=True,
    )

    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    fdd_band_spec = radial_band_spec(256, 256, args.fdd_bands, device)
    real_features, real_stats, real_fdd_mean, real_fdd_std, real_cache = _load_or_create_real_features_and_fdd(
        args=args,
        extractor=extractor,
        device=device,
        output_dir=output_dir,
        fdd_band_spec=fdd_band_spec,
    )
    print(f"real_features_cache={real_cache} features={real_features.shape}", flush=True)
    if args.prepare_real_features_only:
        return

    model = load_sit_xl_2(args.checkpoint, device)
    vae = load_autoencoder_kl(args.vae_dir, device)
    shape = MODEL_SHAPES["sit_xl_2"]
    noise_path = args.noise_bank or _default_noise_bank_path("sit_xl_2", args.seed, args.num_samples, shape)
    noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=shape, seed=args.seed)
    probe_path = args.probe_noise_bank or _default_noise_bank_path("sit_xl_2_probe", args.seed + 777001, args.num_probe, shape)
    probe_noise = load_or_create_noise_bank(probe_path, num_samples=args.num_probe, shape=shape, seed=args.seed + 777001)
    labels = (torch.arange(args.num_samples, dtype=torch.long) % 1000).contiguous()
    probe_labels = (torch.arange(args.num_probe, dtype=torch.long) % 1000).contiguous()
    latent_band_spec = radial_band_spec(shape[-2], shape[-1], args.freq_bands, device)
    metrics_path = output_dir / "metrics.csv"
    density_path = output_dir / "timegrid_density.csv"
    mechanism_path = output_dir / "mechanism_scores.csv"
    rows: list[dict[str, object]] = [dict(row) for row in _read_csv(metrics_path)]
    density_rows: list[dict[str, object]] = [dict(row) for row in _read_csv(density_path)]
    completed = {(str(row["sampler"]), str(row["grid"]), int(row["nfe"])) for row in rows}
    profile_cache: dict[str, dict[str, np.ndarray]] = {}

    for sampler in args.samplers:
        profile_path = output_dir / f"profile_{sampler}_probe{args.num_probe}_bins{args.time_bins}_bands{args.freq_bands}.npz"
        if profile_path.exists():
            payload = np.load(profile_path, allow_pickle=False)
            profile_cache[sampler] = {
                "local_error": payload["local_error"],
                "curvature": payload["curvature"],
                "velocity_rms": payload["velocity_rms"],
            }
        else:
            print(f"profile_start sampler={sampler}", flush=True)
            profile_cache[sampler] = _profile_maps(
                model,
                probe_noise,
                probe_labels,
                sampler,
                args,
                latent_band_spec,
                device,
                amp_enabled,
            )
            np.savez_compressed(profile_path, **profile_cache[sampler])
            print(f"profile_done sampler={sampler} cache={profile_path}", flush=True)

    for sampler in args.samplers:
        for grid in args.grids:
            for nfe in args.nfe:
                key = (sampler, grid, int(nfe))
                if key in completed:
                    print(f"skip_completed sampler={sampler} grid={grid} nfe={nfe}", flush=True)
                    continue
                if grid == "official" and (sampler, "uniform", int(nfe)) in completed:
                    source = [row for row in rows if row["sampler"] == sampler and row["grid"] == "uniform" and int(row["nfe"]) == int(nfe)][0]
                    copied = dict(source)
                    copied["grid"] = "official"
                    copied["grid_note"] = "official_sit_fixed_ode_grid_equivalent_to_uniform_time_for_linear_flow"
                    rows.append(copied)
                    density, density_source = _density_for_grid(grid, profile_cache[sampler], args)
                    density_rows.extend(_density_rows(grid, sampler, int(nfe), density, density_source))
                    _update_delta_to_best(rows)
                    _write_csv(metrics_path, rows)
                    _write_csv(density_path, density_rows)
                    _write_csv(output_dir / "flow_mechanism_correlations.csv", _correlation_rows(rows))
                    completed.add(key)
                    print(f"copy_official_from_uniform sampler={sampler} nfe={nfe}", flush=True)
                    continue

                intervals = _intervals_for_nfe(sampler, int(nfe))
                effective_nfe = _effective_nfe(sampler, intervals)
                density, density_source = _density_for_grid(grid, profile_cache[sampler], args)
                times = _time_grid_from_density(intervals, density, device)
                start_time = time.time()
                sample_time = 0.0
                features: list[np.ndarray] = []
                grid_images: list[torch.Tensor] = []
                fdd_sums = torch.zeros(args.fdd_bands, device=device)
                fdd_sq_sums = torch.zeros(args.fdd_bands, device=device)
                fdd_count = 0
                total_calls = 0
                total_finite = 0
                total_pixels = 0
                print(f"point_start sampler={sampler} grid={grid} nfe={nfe} intervals={intervals}", flush=True)
                for start in range(0, args.num_samples, args.batch_size):
                    end = min(start + args.batch_size, args.num_samples)
                    noise = noise_bank[start:end].to(device, non_blocking=True)
                    y = labels[start:end].to(device, non_blocking=True)
                    t0 = time.time()
                    latents, calls = _sample_latents(model, noise, y, times, sampler, args.cfg_scale, amp_enabled, device)
                    images = _decode(vae, latents, args.latent_scale_factor, amp_enabled, device)
                    sample_time += time.time() - t0
                    features.append(extractor(images).detach().float().cpu().numpy())
                    fdd_count = _spectral_stats_update(images, fdd_band_spec, fdd_sums, fdd_sq_sums, fdd_count)
                    images_cpu = images.detach().float().cpu()
                    total_calls += calls
                    total_finite += int(torch.isfinite(images_cpu).sum().item())
                    total_pixels += int(images_cpu.numel())
                    remain = args.grid_samples - sum(item.shape[0] for item in grid_images)
                    if remain > 0:
                        grid_images.append(images_cpu[:remain])
                    if end % max(args.batch_size * 50, args.batch_size) == 0 or end == args.num_samples:
                        print(f"sample_progress sampler={sampler} grid={grid} nfe={nfe} samples={end}/{args.num_samples}", flush=True)

                fake_features = np.concatenate(features, axis=0)[: args.num_samples]
                fake_stats = feature_stats(fake_features)
                fid = frechet_distance(real_stats, fake_stats)
                real_sub, fake_sub = _subsample_features(real_features, fake_features, args.metric_subset, args.seed + int(nfe) + len(rows))
                cmmd = _cmmd_rbf(real_sub, fake_sub)
                density_metric, coverage_metric = _density_coverage(real_sub, fake_sub, args.density_k)
                fake_fdd_mean, fake_fdd_std = _spectral_stats_finalize(fdd_sums, fdd_sq_sums, fdd_count)
                fdd = _fdd(real_fdd_mean, real_fdd_std, fake_fdd_mean, fake_fdd_std)
                trajectory_drift = _trajectory_drift(
                    model, probe_noise, probe_labels, sampler, times, args, device, amp_enabled
                )
                high = _high_slice(args.freq_bands)
                profile = profile_cache[sampler]
                weights = density / max(float(density.sum()), 1e-12)
                local_high = np.asarray(profile["local_error"])[:, high].sum(axis=1)
                curvature_high = np.asarray(profile["curvature"])[:, high].sum(axis=1)
                d_flow = np.asarray(profile["velocity_rms"])
                grid_path = grid_dir / f"sit_xl_2_{sampler}_{grid}_nfe{nfe}.png"
                if grid_images:
                    _save_grid(grid_path, torch.cat(grid_images, dim=0), args.grid_nrow)
                row = {
                    "model": "sit_xl_2",
                    "sampler": sampler,
                    "grid": grid,
                    "nfe": int(nfe),
                    "effective_nfe": effective_nfe,
                    "intervals": intervals,
                    "fid": fid,
                    "delta_fid_to_best": float("nan"),
                    "best_observed_fid": float("nan"),
                    "cmmd": cmmd,
                    "density": density_metric,
                    "coverage": coverage_metric,
                    "fdd": fdd,
                    "wall_clock_sec": sample_time,
                    "total_elapsed_sec": time.time() - start_time,
                    "sec_per_sample": sample_time / max(args.num_samples, 1),
                    "finite_fraction": total_finite / max(total_pixels, 1),
                    "trajectory_drift": trajectory_drift,
                    "density_weighted_local_high_error": float((weights * local_high).sum()),
                    "density_weighted_curvature_high": float((weights * curvature_high).sum()),
                    "density_weighted_d_flow": float((weights * d_flow).sum()),
                    "density_min": float(density.min()),
                    "density_max": float(density.max()),
                    "density_argmax_bin": int(density.argmax()),
                    "density_source": density_source,
                    "profile_strength": args.profile_strength,
                    "max_density_ratio": args.max_density_ratio,
                    "num_samples": args.num_samples,
                    "num_real_stats": args.num_real_stats,
                    "metric_subset": min(args.metric_subset, len(real_sub), len(fake_sub)),
                    "density_k": args.density_k,
                    "num_probe": args.num_probe,
                    "ref_substeps": args.ref_substeps,
                    "cfg_scale": args.cfg_scale,
                    "checkpoint": args.checkpoint,
                    "vae_dir": args.vae_dir,
                    "noise_bank_id": noise_bank_id(noise_path),
                    "probe_noise_bank_id": noise_bank_id(probe_path),
                    "real_features_cache": str(real_cache),
                    "grid_path": str(grid_path),
                    "metric_note": "fid10k_inception; cmmd/density/coverage use approximate feature subset; fdd uses radial image-spectrum moments",
                    "grid_note": "official grid equals uniform time for SiT linear ODE" if grid == "official" else "",
                }
                rows.append(row)
                density_rows.extend(_density_rows(grid, sampler, int(nfe), density, density_source))
                _update_delta_to_best(rows)
                _write_csv(metrics_path, rows)
                _write_csv(density_path, density_rows)
                _write_csv(mechanism_path, rows)
                _write_csv(output_dir / "flow_mechanism_correlations.csv", _correlation_rows(rows))
                completed.add(key)
                print(
                    f"point_done sampler={sampler} grid={grid} nfe={nfe} fid={fid:.4f} cmmd={cmmd:.6f} "
                    f"density={density_metric:.4f} coverage={coverage_metric:.4f} fdd={fdd:.6f} "
                    f"drift={trajectory_drift:.6f} sec={sample_time:.2f}",
                    flush=True,
                )

    _update_delta_to_best(rows)
    _write_csv(metrics_path, rows)
    _write_csv(mechanism_path, rows)
    _write_csv(density_path, density_rows)
    _write_csv(output_dir / "flow_mechanism_correlations.csv", _correlation_rows(rows))
    print(f"flow_validation_done metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
