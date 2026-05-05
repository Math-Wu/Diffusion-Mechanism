from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.imagenet256 import decode_latents, imagenet256_schedule, load_autoencoder_kl, load_imagenet256_model
from dm.eval_utils import load_or_create_noise_bank, noise_bank_id
from dm.metrics import FeatureStats, build_feature_extractor, feature_stats, frechet_distance
from dm.samplers.base import SamplerResult
from dm.samplers.ode import AB_COEFFS
from dm.utils import default_device, ensure_dir, set_seed


SOLVERS = ("ddim", "heun", "dpmpp", "unipc")
MODEL_SHAPES = {
    "adm256": (3, 256, 256),
    "dit_xl_2": (4, 32, 32),
    "uvit_l_2": (4, 32, 32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet256 pretrained sampler sweep for ADM, DiT, and U-ViT.")
    parser.add_argument("--model", choices=sorted(MODEL_SHAPES), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae_dir", help="Required for latent-space models.")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=SOLVERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--num_real_stats", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--real_npz", default="data/imagenet256/ref_batches/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--stats_only", action="store_true")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    parser.add_argument("--delta_reference", default="best_observed", choices=["best_observed", "none"])
    parser.add_argument("--grid_samples", type=int, default=64)
    parser.add_argument("--grid_nrow", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent_scale_factor", type=float, default=0.18215)
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _macro_intervals(solver: str, nfe: int) -> int:
    return max(1, (nfe + 1) // 2) if solver == "heun" else max(1, nfe)


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
def _heun_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    *,
    full_corrector: bool,
) -> tuple[torch.Tensor, int]:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t, y)
    drift = schedule.drift(x, t, eps)
    if not full_corrector:
        return x + dt * drift, 1
    x_euler = x + dt * drift
    eps_next = model(x_euler, t_next, y)
    drift_next = schedule.drift(x_euler, t_next, eps_next)
    return x + 0.5 * dt * (drift + drift_next), 2


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
def sample_grid(model, schedule, noise: torch.Tensor, labels: torch.Tensor, solver: str, nfe: int) -> SamplerResult:
    intervals = _macro_intervals(solver, nfe)
    times = schedule.time_grid(intervals, noise.device)
    x = noise
    history: list[torch.Tensor] = []
    calls = 0
    if solver == "heun":
        remaining = nfe
        for index in range(times.numel() - 1):
            t = _batch_time(times[index], x.shape[0])
            t_next = _batch_time(times[index + 1], x.shape[0])
            full_corrector = remaining > 1
            x, used = _heun_step(model, schedule, x, t, t_next, labels, full_corrector=full_corrector)
            remaining -= used
            calls += used
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


def _save_grid(path: Path, images: torch.Tensor, nrow: int) -> None:
    try:
        from torchvision.utils import save_image
    except Exception as exc:
        print(f"grid_save_skip path={path} error={exc}", flush=True)
        return
    save_image((images.clamp(-1, 1) + 1.0) * 0.5, path, nrow=nrow)


def _default_noise_bank_path(model_name: str, seed: int, num_samples: int, shape: tuple[int, int, int]) -> Path:
    shape_tag = "x".join(str(value) for value in shape)
    return Path("data/noise_banks") / f"imagenet256_{model_name}_seed{seed}_n{num_samples}_{shape_tag}.pt"


def _default_real_stats_path(feature_backend: str, num_samples: int) -> Path:
    return Path("data/fid_stats") / f"imagenet256_val_{feature_backend}_n{num_samples}.npz"


def _real_image_batches_from_npz(npz_path: str | Path, total: int, batch_size: int):
    data = np.load(npz_path, allow_pickle=False)
    if "arr_0" not in data:
        raise ValueError(f"{npz_path} does not contain arr_0 real image array")
    images = data["arr_0"]
    if images.ndim != 4:
        raise ValueError(f"Expected arr_0 to be rank-4, got shape {images.shape}")
    count = min(total, images.shape[0])
    for start in range(0, count, batch_size):
        batch = images[start : min(start + batch_size, count)]
        tensor = torch.from_numpy(np.ascontiguousarray(batch))
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.float().div(127.5).sub(1.0)
        yield tensor


def _load_imagenet256_real_stats(
    *,
    npz_path: str | Path,
    cache_path: str | Path,
    extractor: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    feature_backend: str,
) -> FeatureStats:
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return FeatureStats(mu=data["mu"], sigma=data["sigma"])
    features: list[np.ndarray] = []
    seen = 0
    for batch in _real_image_batches_from_npz(npz_path, num_samples, batch_size):
        batch = batch.to(device, non_blocking=True)
        features.append(extractor(batch).detach().float().cpu().numpy())
        seen += batch.shape[0]
        print(f"real_stats_progress {seen}/{num_samples}", flush=True)
    stats = feature_stats(np.concatenate(features, axis=0)[:num_samples])
    ensure_dir(cache_path.parent)
    np.savez(
        cache_path,
        mu=stats.mu,
        sigma=stats.sigma,
        dataset="imagenet256_val",
        real_npz=str(npz_path),
        feature_backend=feature_backend,
        num_samples=num_samples,
    )
    return stats


def _update_delta_fid(rows: list[dict[str, object]], mode: str) -> None:
    if not rows:
        return
    if mode == "none":
        for row in rows:
            row["delta_fid"] = float("nan")
            row["reference_fid"] = float("nan")
            row["delta_reference"] = "none"
        return
    best_fid = min(float(row["fid"]) for row in rows)
    for row in rows:
        row["delta_fid"] = float(row["fid"]) - best_fid
        row["reference_fid"] = best_fid
        row["delta_reference"] = "best_observed_same_architecture"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    grid_dir = ensure_dir(output_dir / "sample_grids")
    print(f"imagenet256_sweep_start model={args.model} device={device} output_dir={output_dir}", flush=True)
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    num_real_stats = args.num_real_stats or args.num_samples
    real_stats_path = args.real_stats_cache or _default_real_stats_path(args.feature_backend, num_real_stats)
    real_stats = _load_imagenet256_real_stats(
        npz_path=args.real_npz,
        cache_path=real_stats_path,
        extractor=extractor,
        device=device,
        num_samples=num_real_stats,
        batch_size=args.real_batch_size,
        feature_backend=args.feature_backend,
    )
    print(f"real_stats_cache={real_stats_path} num_real_stats={num_real_stats}", flush=True)
    if args.stats_only:
        return

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
    noise_id = noise_bank_id(noise_path)
    labels = (torch.arange(args.num_samples, dtype=torch.long) % 1000).contiguous()
    rows: list[dict[str, object]] = []
    amp_enabled = bool(args.amp and device.type == "cuda")

    for solver in args.solvers:
        for nfe in args.nfe:
            print(f"sweep_point_start model={args.model} solver={solver} nfe={nfe}", flush=True)
            start_time = time.time()
            total_calls = 0
            total_finite = 0
            total_pixels = 0
            pixel_sum = 0.0
            pixel_sq_sum = 0.0
            pixel_min = float("inf")
            pixel_max = float("-inf")
            sampling_runtime = 0.0
            features: list[np.ndarray] = []
            grid_images: list[torch.Tensor] = []
            for start in range(0, args.num_samples, args.batch_size):
                end = min(start + args.batch_size, args.num_samples)
                noise = noise_bank[start:end].to(device, non_blocking=True)
                y = labels[start:end].to(device, non_blocking=True)
                sample_start = time.time()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    result = sample_grid(model, schedule, noise, y, solver, nfe)
                    samples = result.samples
                    if vae is not None:
                        samples = decode_latents(vae, samples, scale_factor=args.latent_scale_factor)
                sampling_runtime += time.time() - sample_start
                samples = samples.detach().float()
                features.append(extractor(samples).detach().float().cpu().numpy())
                samples_f = samples.cpu()
                total_calls += result.nfe
                total_finite += int(torch.isfinite(samples_f).sum().item())
                total_pixels += int(samples_f.numel())
                pixel_sum += float(samples_f.sum().item())
                pixel_sq_sum += float(samples_f.square().sum().item())
                pixel_min = min(pixel_min, float(samples_f.min().item()))
                pixel_max = max(pixel_max, float(samples_f.max().item()))
                remaining_grid = args.grid_samples - sum(item.shape[0] for item in grid_images)
                if remaining_grid > 0:
                    grid_images.append(samples_f[:remaining_grid])
            elapsed = time.time() - start_time
            mean = pixel_sum / max(total_pixels, 1)
            variance = max(pixel_sq_sum / max(total_pixels, 1) - mean * mean, 0.0)
            finite_fraction = total_finite / max(total_pixels, 1)
            fid = frechet_distance(real_stats, feature_stats(np.concatenate(features, axis=0)[: args.num_samples]))
            grid_path = grid_dir / f"{args.model}_{solver}_nfe{nfe}.png"
            if grid_images:
                _save_grid(grid_path, torch.cat(grid_images, dim=0), args.grid_nrow)
            row = {
                "architecture": args.model,
                "model": args.model,
                "solver": solver,
                "nfe": nfe,
                "fid": fid,
                "delta_fid": float("nan"),
                "reference_fid": float("nan"),
                "delta_reference": args.delta_reference,
                "num_samples": args.num_samples,
                "num_real_stats": num_real_stats,
                "batch_size": args.batch_size,
                "wall_clock_sec": sampling_runtime,
                "total_elapsed_sec": elapsed,
                "sec_per_sample": sampling_runtime / max(args.num_samples, 1),
                "nfe_per_sample": nfe,
                "model_call_batches": total_calls,
                "finite_fraction": finite_fraction,
                "pixel_mean": mean,
                "pixel_std": variance ** 0.5,
                "pixel_min": pixel_min,
                "pixel_max": pixel_max,
                "checkpoint": args.checkpoint,
                "vae_dir": args.vae_dir or "",
                "seed": args.seed,
                "noise_bank_id": noise_id,
                "noise_bank": str(noise_path),
                "real_npz": str(args.real_npz),
                "real_stats_cache": str(real_stats_path),
                "feature_backend": args.feature_backend,
                "grid_path": str(grid_path),
                "metric_note": "imagenet256_pretrained_torchvision_inception_proxy_fid",
            }
            rows.append(row)
            _update_delta_fid(rows, args.delta_reference)
            _write_csv(output_dir / "metrics.csv", rows)
            print(
                f"sweep_point_done model={args.model} solver={solver} nfe={nfe} "
                f"fid={fid:.4f} delta={row['delta_fid']:.4f} sample_sec={sampling_runtime:.2f} "
                f"total_sec={elapsed:.2f} finite={finite_fraction:.6f} mean={mean:.4f} std={row['pixel_std']:.4f}",
                flush=True,
            )
    print(output_dir / "metrics.csv", flush=True)


if __name__ == "__main__":
    main()
