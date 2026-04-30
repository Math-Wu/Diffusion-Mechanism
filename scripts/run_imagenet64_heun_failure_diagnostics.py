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
from dm.eval_utils import load_or_compute_real_stats
from dm.imagenet64 import build_imagenet64_val_loader, load_imagenet64_pretrained_model
from dm.metrics import build_feature_extractor, feature_stats, frechet_distance
from dm.utils import default_device, ensure_dir, set_seed
from run_imagenet64_external_smoke import (
    LabelConditionedModel,
    _batch_time,
    _difficulty_map,
    _heun_step,
    _high_band_slice,
    _limited_batches,
    _macro_intervals,
    _normalize_profile,
    _reference_heun,
    _sample_grid,
    _schedule_for_architecture,
    _time_grid_from_density,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ImageNet64 U-ViT Heun p0_high failure modes.")
    parser.add_argument("--data_npz", default="data/imagenet64/Imagenet64_val_npz/val_data.npz")
    parser.add_argument("--uvit_checkpoint", default="checkpoints/u-vit/imagenet64_uvit_large.pth")
    parser.add_argument("--output_dir", default="outputs/imagenet64_uvit_heun_failure_diagnostics")
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 7, 8])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5])
    parser.add_argument("--cap_strengths", nargs="+", type=float, default=[0.2, 0.35, 0.5])
    parser.add_argument("--bin0_caps", nargs="+", type=float, default=[1.1, 1.2, 1.35])
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_real_stats", type=int, default=50000)
    parser.add_argument("--num_probe", type=int, default=256)
    parser.add_argument("--num_p2_paths", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--max_density_ratio", type=float, default=2.5)
    parser.add_argument("--real_stats_cache", default="outputs/imagenet64_external_formal/imagenet64_val_inception_n50000.npz")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def _euler_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
    return x + dt * drift


def _heun_full_corrector_count(nfe: int) -> int:
    return nfe // 2


def _heun_interval_is_full_corrector(nfe: int, interval_index: int) -> bool:
    return interval_index < _heun_full_corrector_count(nfe)


@torch.no_grad()
def _heun_actual_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    *,
    full_corrector: bool,
) -> torch.Tensor:
    if full_corrector:
        return _heun_step(model, schedule, x, t, t_next)
    return _euler_step(model, schedule, x, t, t_next)


@torch.no_grad()
def _heun_error_map_actual(
    model,
    schedule,
    loader,
    nfe: int,
    total: int,
    time_bins: int,
    band_spec,
    ref_substeps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = logsnr_bin_edges(schedule, time_bins, device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    intervals = _macro_intervals("heun", nfe)
    lambda_step = (edges[-1] - edges[0]) / intervals
    interval_indices = torch.clamp(torch.floor((centers - edges[0]) / lambda_step), 0, intervals - 1).long()
    full_corrector = torch.tensor(
        [_heun_interval_is_full_corrector(nfe, int(index.item())) for index in interval_indices],
        device=device,
        dtype=torch.bool,
    )

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
            x_next = _heun_actual_step(
                conditioned,
                schedule,
                x_start,
                t,
                t_next,
                full_corrector=bool(full_corrector[bin_index].item()),
            )
            sums[bin_index] += fft_radial_band_energy(x_next - x_ref, band_spec).sum(dim=0)
            counts[bin_index] += x0.shape[0]
    error_map = (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()
    return error_map, interval_indices.detach().cpu().numpy(), full_corrector.detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def _p2_heun_actual_many(
    model,
    schedule,
    labels: torch.Tensor,
    noise: torch.Tensor,
    nfe: int,
    band_spec,
    ref_substeps: int,
    batch_size: int,
    num_paths: int,
    device: torch.device,
) -> dict[str, float]:
    intervals = _macro_intervals("heun", nfe)
    times = schedule.time_grid(intervals, device)
    high = _high_band_slice(band_spec.masks.shape[0])
    x0_high = 0.0
    drift = 0.0
    count = min(num_paths, labels.shape[0], noise.shape[0])
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch_labels = labels[start:end].to(device, non_blocking=True)
        x_solver = noise[start:end].to(device, non_blocking=True).clone()
        x_ref = x_solver.clone()
        conditioned = LabelConditionedModel(model, batch_labels)
        for index in range(times.numel() - 1):
            t = _batch_time(times[index], x_solver.shape[0])
            t_next = _batch_time(times[index + 1], x_solver.shape[0])
            x_solver = _heun_actual_step(
                conditioned,
                schedule,
                x_solver,
                t,
                t_next,
                full_corrector=_heun_interval_is_full_corrector(nfe, index),
            )
            x_ref = _reference_heun(conditioned, schedule, x_ref, t, t_next, ref_substeps)
            eps_solver = conditioned(x_solver, t_next)
            eps_ref = conditioned(x_ref, t_next)
            x0_solver = schedule.eps_to_x0(x_solver, t_next, eps_solver)
            x0_ref = schedule.eps_to_x0(x_ref, t_next, eps_ref)
            x0_high += float(fft_radial_band_energy(x0_solver - x0_ref, band_spec)[:, high].sum().detach().cpu())
            drift += float((x_solver - x_ref).flatten(1).square().mean(dim=1).sqrt().sum().detach().cpu())
    return {
        "p2_x0_high_error": x0_high / max(count, 1),
        "p2_trajectory_drift": drift / max(count, 1),
    }


def _cap_density_bin0(density: np.ndarray, cap: float) -> np.ndarray:
    adjusted = np.asarray(density, dtype=np.float64).copy()
    if adjusted.size <= 1 or adjusted[0] <= cap:
        return adjusted / max(float(adjusted.mean()), 1e-12)
    excess = adjusted[0] - cap
    adjusted[0] = cap
    tail_sum = float(adjusted[1:].sum())
    if tail_sum > 1e-12:
        adjusted[1:] += excess * adjusted[1:] / tail_sum
    return adjusted / max(float(adjusted.mean()), 1e-12)


def _density_no_bin0(profile: np.ndarray, strength: float, max_density_ratio: float) -> np.ndarray:
    no_bin0 = np.asarray(profile, dtype=np.float64).copy()
    no_bin0[0] = 0.0
    density = _normalize_profile(no_bin0, strength, max_density_ratio)
    return _cap_density_bin0(density, 1.0)


def _density_rows(
    schedule,
    nfe: int,
    mode: str,
    strength: float,
    density: np.ndarray,
    *,
    cap_bin0: float | None = None,
) -> list[dict[str, object]]:
    edges = logsnr_bin_edges(schedule, len(density), torch.device("cpu")).double()
    centers = 0.5 * (edges[:-1] + edges[1:])
    t_centers = schedule.inverse_log_snr(centers.float()).double()
    rows: list[dict[str, object]] = []
    for bin_index, value in enumerate(density):
        rows.append(
            {
                "architecture": "uvit",
                "solver": "heun",
                "nfe": nfe,
                "mode": mode,
                "strength": strength,
                "cap_bin0": "" if cap_bin0 is None else cap_bin0,
                "bin": bin_index,
                "logsnr_left": float(edges[bin_index].item()),
                "logsnr_right": float(edges[bin_index + 1].item()),
                "logsnr_center": float(centers[bin_index].item()),
                "t_center": float(t_centers[bin_index].item()),
                "density": float(value),
            }
        )
    return rows


def _gap_rows(schedule, nfe: int, mode: str, strength: float, times: torch.Tensor, *, cap_bin0: float | None = None) -> list[dict[str, object]]:
    times_cpu = times.detach().float().cpu()
    lambdas = schedule.log_snr(times_cpu).detach().float().cpu()
    rows: list[dict[str, object]] = []
    for step in range(times_cpu.numel() - 1):
        rows.append(
            {
                "architecture": "uvit",
                "solver": "heun",
                "nfe": nfe,
                "mode": mode,
                "strength": strength,
                "cap_bin0": "" if cap_bin0 is None else cap_bin0,
                "macro_step": step,
                "full_corrector": int(_heun_interval_is_full_corrector(nfe, step)),
                "t_start": float(times_cpu[step].item()),
                "t_end": float(times_cpu[step + 1].item()),
                "t_gap": float((times_cpu[step] - times_cpu[step + 1]).item()),
                "logsnr_start": float(lambdas[step].item()),
                "logsnr_end": float(lambdas[step + 1].item()),
                "logsnr_gap": float((lambdas[step + 1] - lambdas[step]).item()),
            }
        )
    return rows


def _sample_features(
    model,
    schedule,
    extractor,
    labels_all: torch.Tensor,
    noise_all: torch.Tensor,
    nfe: int,
    times: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    features = []
    total_nfe = 0
    for start in range(0, noise_all.shape[0], batch_size):
        end = min(start + batch_size, noise_all.shape[0])
        result = _sample_grid(
            model,
            schedule,
            noise_all[start:end].to(device, non_blocking=True),
            labels_all[start:end].to(device, non_blocking=True),
            "heun",
            nfe,
            times,
        )
        total_nfe += result.nfe
        features.append(extractor(result.samples).detach().float().cpu().numpy())
    return np.concatenate(features, axis=0)[: noise_all.shape[0]], total_nfe


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)

    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    real_loader = build_imagenet64_val_loader(args.data_npz, args.real_batch_size, num_workers=0, shuffle=False)
    real_stats = load_or_compute_real_stats(
        args.real_stats_cache,
        _limited_batches(real_loader, args.num_real_stats, device),
        extractor,
        device,
        args.num_real_stats,
        metadata={"dataset": "imagenet64_val", "feature_backend": args.feature_backend, "num_real_stats": args.num_real_stats},
    )
    print(f"real_stats_cache={args.real_stats_cache} num_real_stats={args.num_real_stats}", flush=True)

    model = load_imagenet64_pretrained_model("uvit", args.uvit_checkpoint, device)
    schedule = _schedule_for_architecture("uvit")
    band_spec = radial_band_spec(64, 64, args.freq_bands, device)
    probe_loader = build_imagenet64_val_loader(args.data_npz, args.batch_size, num_workers=0, shuffle=False)
    sample_loader = build_imagenet64_val_loader(args.data_npz, args.batch_size, num_workers=0, shuffle=False)

    difficulty = _difficulty_map(model, schedule, probe_loader, args.num_probe, args.time_bins, band_spec, device)
    difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)

    labels_all = []
    for _x, y in itertools.islice(sample_loader, (args.num_samples + args.batch_size - 1) // args.batch_size):
        labels_all.append(y)
    labels_all = torch.cat(labels_all, dim=0)[: args.num_samples].contiguous()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_all = torch.randn(args.num_samples, 3, 64, 64, generator=generator)

    metrics_rows: list[dict[str, object]] = []
    p0_rows: list[dict[str, object]] = []
    p2_rows: list[dict[str, object]] = []
    density_csv_rows: list[dict[str, object]] = []
    gap_csv_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    maps_payload: dict[str, np.ndarray] = {
        "difficulty": difficulty,
        "difficulty_norm": difficulty_norm,
    }

    for nfe in args.nfe:
        error_map, interval_indices, full_corrector = _heun_error_map_actual(
            model,
            schedule,
            probe_loader,
            nfe,
            args.num_probe,
            args.time_bins,
            band_spec,
            args.ref_substeps,
            device,
        )
        p0_high = (difficulty_norm * error_map)[:, _high_band_slice(args.freq_bands)].sum(axis=1)
        maps_payload[f"error_map_nfe_{nfe}"] = error_map
        maps_payload[f"p0_high_nfe_{nfe}"] = p0_high
        maps_payload[f"interval_index_nfe_{nfe}"] = interval_indices
        maps_payload[f"full_corrector_nfe_{nfe}"] = full_corrector
        for bin_index, (interval_index, is_full) in enumerate(zip(interval_indices, full_corrector)):
            bin_rows.append(
                {
                    "architecture": "uvit",
                    "solver": "heun",
                    "nfe": nfe,
                    "bin": bin_index,
                    "macro_interval": int(interval_index),
                    "full_corrector": int(is_full),
                    "p0_high": float(p0_high[bin_index]),
                    "difficulty_sum": float(difficulty[bin_index].sum()),
                    "error_sum": float(error_map[bin_index].sum()),
                }
            )
        p0_rows.append(
            {
                "architecture": "uvit",
                "solver": "heun",
                "nfe": nfe,
                "difficulty_weighted_error": float((difficulty_norm * error_map).sum()),
                "p0_high_sum": float(p0_high.sum()),
                "total_error": float(error_map.sum()),
                "num_probe": args.num_probe,
                "ref_substeps": args.ref_substeps,
                "time_bins": args.time_bins,
                "freq_bands": args.freq_bands,
                "argmax_bin": int(p0_high.argmax()),
                "argmax_full_corrector": int(full_corrector[int(p0_high.argmax())]),
            }
        )
        p2 = _p2_heun_actual_many(
            model,
            schedule,
            labels_all,
            noise_all,
            nfe,
            band_spec,
            args.ref_substeps,
            args.batch_size,
            args.num_p2_paths,
            device,
        )
        p2_rows.append(
            {
                "architecture": "uvit",
                "solver": "heun",
                "nfe": nfe,
                "num_p2_paths": args.num_p2_paths,
                "ref_substeps": args.ref_substeps,
                **p2,
            }
        )

        intervals = _macro_intervals("heun", nfe)
        candidates: list[tuple[str, float, np.ndarray, float | None]] = [
            ("uniform", 0.0, np.ones(args.time_bins, dtype=np.float64), None)
        ]
        for strength in args.strengths:
            candidates.append(("p0_high", strength, _normalize_profile(p0_high, strength, args.max_density_ratio), None))
        for strength in args.cap_strengths:
            base_density = _normalize_profile(p0_high, strength, args.max_density_ratio)
            candidates.append(("p0_high_no_bin0", strength, _density_no_bin0(p0_high, strength, args.max_density_ratio), None))
            for cap in args.bin0_caps:
                candidates.append(("p0_high_cap_bin0", strength, _cap_density_bin0(base_density, cap), cap))

        baseline_fid = None
        for mode, strength, density, cap in candidates:
            times = _time_grid_from_density(schedule, intervals, density, device)
            density_csv_rows.extend(_density_rows(schedule, nfe, mode, strength, density, cap_bin0=cap))
            gap_csv_rows.extend(_gap_rows(schedule, nfe, mode, strength, times, cap_bin0=cap))
            features, total_nfe = _sample_features(
                model,
                schedule,
                extractor,
                labels_all,
                noise_all,
                nfe,
                times,
                args.batch_size,
                device,
            )
            fid = frechet_distance(real_stats, feature_stats(features))
            if mode == "uniform":
                baseline_fid = fid
            delta = fid - baseline_fid if baseline_fid is not None else 0.0
            metrics_rows.append(
                {
                    "architecture": "uvit",
                    "solver": "heun",
                    "nfe": nfe,
                    "mode": mode,
                    "strength": strength,
                    "cap_bin0": "" if cap is None else cap,
                    "fid": fid,
                    "delta_vs_uniform": delta,
                    "num_samples": args.num_samples,
                    "num_real_stats": args.num_real_stats,
                    "num_probe": args.num_probe,
                    "num_p2_paths": args.num_p2_paths,
                    "ref_substeps": args.ref_substeps,
                    "time_bins": args.time_bins,
                    "freq_bands": args.freq_bands,
                    "max_density_ratio": args.max_density_ratio,
                    "density_min": float(density.min()),
                    "density_max": float(density.max()),
                    "density_argmax_bin": int(density.argmax()),
                    "density_bin0": float(density[0]),
                    "total_model_calls": total_nfe,
                    "real_stats_cache": args.real_stats_cache,
                    "feature_backend": args.feature_backend,
                }
            )
            print(
                f"uvit heun nfe={nfe} {mode} strength={strength:g} cap={'' if cap is None else cap:g} "
                f"fid={fid:.4f} delta={delta:.4f}",
                flush=True,
            )
            _write_csv(output_dir / "heun_failure_metrics.csv", metrics_rows)
        _write_csv(output_dir / "heun_failure_p0_scores.csv", p0_rows)
        _write_csv(output_dir / "heun_failure_p2_scores.csv", p2_rows)
        _write_csv(output_dir / "heun_failure_bin_diagnostics.csv", bin_rows)
        _write_csv(output_dir / "heun_failure_timegrid_density.csv", density_csv_rows)
        _write_csv(output_dir / "heun_failure_timegrid_gaps.csv", gap_csv_rows)
        np.savez(output_dir / "heun_failure_maps.npz", **maps_payload)

    print(output_dir / "heun_failure_metrics.csv", flush=True)
    print(output_dir / "heun_failure_p0_scores.csv", flush=True)
    print(output_dir / "heun_failure_timegrid_density.csv", flush=True)
    print(output_dir / "heun_failure_timegrid_gaps.csv", flush=True)
    print(output_dir / "heun_failure_maps.npz", flush=True)


if __name__ == "__main__":
    main()
