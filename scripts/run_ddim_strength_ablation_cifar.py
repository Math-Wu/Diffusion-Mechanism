from __future__ import annotations

import argparse
import csv
import itertools
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
)
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.metrics import build_feature_extractor, feature_stats, frechet_distance
from dm.utils import default_device, ensure_dir, set_seed
from run_timegrid_intervention_cifar import (
    generated_feature_stats,
    profile_from_maps,
    time_grid_from_density,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDIM-only p0_high strength ablation using existing uniform baselines."
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--baseline_csvs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/ddim_p0_high_strength_ablation_10k")
    parser.add_argument("--p0_maps", default="outputs/mechanism_p0_cifar_medium_fixedbin/mechanism_maps.npz")
    parser.add_argument("--p2_maps", default="outputs/trajectory_mechanisms_cifar_medium/trajectory_maps.npz")
    parser.add_argument("--nfe", nargs="+", type=int, default=[8, 15, 20])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.1, 0.2, 0.35])
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint_name", default="last.pt")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    parser.add_argument("--noise_bank")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--max_density_ratio", type=float, default=2.5)
    parser.add_argument("--smooth_profile", action="store_true")
    parser.add_argument("--raw_weights", action="store_true")
    return parser.parse_args()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_uniform_baselines(paths: list[str], num_samples: int) -> dict[tuple[str, int], float]:
    baselines: dict[tuple[str, int], float] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("solver") != "ddim" or row.get("intervention_mode") != "uniform":
                    continue
                if int(row.get("num_samples", num_samples)) != num_samples:
                    continue
                key = (str(row["architecture"]), int(row["nfe"]))
                baselines[key] = float(row["fid"])
    return baselines


def _record_grid_rows(
    rows: list[dict[str, object]],
    *,
    schedule,
    architecture: str,
    nfe: int,
    strength: float,
    density: np.ndarray,
    times: torch.Tensor,
) -> None:
    edges = logsnr_bin_edges(schedule, len(density), torch.device("cpu")).double()
    centers = 0.5 * (edges[:-1] + edges[1:])
    t_centers = schedule.inverse_log_snr(centers.float()).double()
    for bin_index, value in enumerate(density):
        rows.append(
            {
                "architecture": architecture,
                "nfe": nfe,
                "profile_strength": strength,
                "bin": bin_index,
                "logsnr_left": float(edges[bin_index].item()),
                "logsnr_right": float(edges[bin_index + 1].item()),
                "logsnr_center": float(centers[bin_index].item()),
                "t_center": float(t_centers[bin_index].item()),
                "density": float(value),
            }
        )


def _record_gap_rows(
    rows: list[dict[str, object]],
    *,
    schedule,
    architecture: str,
    nfe: int,
    strength: float,
    times: torch.Tensor,
) -> None:
    times_cpu = times.detach().float().cpu()
    lambdas = schedule.log_snr(times_cpu).detach().float().cpu()
    for step in range(nfe):
        rows.append(
            {
                "architecture": architecture,
                "nfe": nfe,
                "profile_strength": strength,
                "step": step,
                "t_start": float(times_cpu[step].item()),
                "t_end": float(times_cpu[step + 1].item()),
                "t_gap": float((times_cpu[step] - times_cpu[step + 1]).item()),
                "logsnr_start": float(lambdas[step].item()),
                "logsnr_end": float(lambdas[step + 1].item()),
                "logsnr_gap": float((lambdas[step + 1] - lambdas[step]).item()),
            }
        )


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.run_dirs):
        raise ValueError("--configs and --run_dirs must have the same length")

    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "ddim_strength_metrics.csv"
    density_path = output_dir / "ddim_timegrid_density.csv"
    gaps_path = output_dir / "ddim_timegrid_gaps.csv"

    baselines = _read_uniform_baselines(args.baseline_csvs, args.num_samples)
    p0_maps = np.load(args.p0_maps)
    p2_maps = np.load(args.p2_maps)
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)

    metrics_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    real_stats_cache = None
    real_stats_path = None
    noise_bank = None
    noise_path = None
    noise_id = None

    for config_path, run_dir in zip(args.configs, args.run_dirs):
        checkpoint = checkpoint_path_for_run(run_dir, name=args.checkpoint_name)
        model, config, ckpt_payload = load_model_from_checkpoint(
            config_path, checkpoint, device, use_ema=not args.raw_weights
        )
        config["training"]["batch_size"] = args.batch_size
        schedule = build_schedule(config)
        architecture = str(config["model"]["architecture"])
        image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
        noise_shape = (int(config["model"]["in_channels"]), image_size, image_size)

        _, val_loader = __import__("dm.data", fromlist=["build_cifar10_loaders"]).build_cifar10_loaders(
            config, download=False
        )
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
            noise_bank = load_or_create_noise_bank(
                noise_path, num_samples=args.num_samples, shape=noise_shape, seed=args.seed
            )
            noise_id = noise_bank_id(noise_path)

        checkpoint_id = f"step_{int(ckpt_payload.get('step', -1))}_images_{int(ckpt_payload.get('images_seen', -1))}"

        for nfe in args.nfe:
            baseline_key = (architecture, nfe)
            if baseline_key not in baselines:
                raise KeyError(f"Missing DDIM uniform baseline for {architecture} nfe={nfe}")
            baseline_fid = baselines[baseline_key]

            uniform_times = time_grid_from_density(
                schedule, nfe, np.ones(int(p0_maps["difficulty"].shape[1]), dtype=np.float64), device
            )
            _record_gap_rows(
                gap_rows,
                schedule=schedule,
                architecture=architecture,
                nfe=nfe,
                strength=0.0,
                times=uniform_times,
            )

            for strength in args.strengths:
                density = profile_from_maps(
                    p0_maps,
                    p2_maps,
                    architecture,
                    "ddim",
                    nfe,
                    "p0_high",
                    strength=strength,
                    max_density_ratio=args.max_density_ratio,
                    smooth=args.smooth_profile,
                )
                times = time_grid_from_density(schedule, nfe, density, device)
                _record_grid_rows(
                    density_rows,
                    schedule=schedule,
                    architecture=architecture,
                    nfe=nfe,
                    strength=strength,
                    density=density,
                    times=times,
                )
                _record_gap_rows(
                    gap_rows,
                    schedule=schedule,
                    architecture=architecture,
                    nfe=nfe,
                    strength=strength,
                    times=times,
                )

                stats, runtime, total_nfe = generated_feature_stats(
                    model,
                    schedule,
                    extractor,
                    "ddim",
                    nfe,
                    times,
                    args.batch_size,
                    noise_bank,
                    args.num_samples,
                    device,
                )
                fid = frechet_distance(real_stats_cache, stats)
                row = {
                    "architecture": architecture,
                    "solver": "ddim",
                    "nfe": nfe,
                    "intervention_mode": "p0_high",
                    "fid": fid,
                    "uniform_fid": baseline_fid,
                    "delta_vs_uniform": fid - baseline_fid,
                    "wall_clock_sec": runtime,
                    "total_model_calls": total_nfe,
                    "num_samples": args.num_samples,
                    "profile_strength": strength,
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
                metrics_rows.append(row)
                _write_rows(metrics_path, metrics_rows)
                _write_rows(density_path, density_rows)
                _write_rows(gaps_path, gap_rows)
                print(
                    f"{architecture} ddim nfe={nfe} p0_high strength={strength:g} "
                    f"fid={fid:.4f} delta_vs_uniform={row['delta_vs_uniform']:.4f}",
                    flush=True,
                )

    print(metrics_path, flush=True)
    print(density_path, flush=True)
    print(gaps_path, flush=True)


if __name__ == "__main__":
    main()
