from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import logsnr_bin_edges, radial_band_spec
from dm.cifar_public import build_public_cifar_model, public_cifar_schedule
from dm.data import build_cifar10_loaders
from dm.eval_utils import (
    default_noise_bank_path,
    default_real_stats_path,
    load_or_compute_real_stats,
    load_or_create_noise_bank,
    noise_bank_id,
)
from dm.metrics import build_feature_extractor, feature_stats, frechet_distance
from dm.utils import default_device, ensure_dir, set_seed
from run_imagenet64_external_smoke import (
    _difficulty_map,
    _high_band_slice,
    _normalize_profile,
    _sample_grid,
    _solver_error_map,
    _time_grid_from_density,
)


MODEL_DEFAULTS = {
    "public_ddpm_unet": "checkpoints/ddpm-cifar10-32",
    "public_uvit": "checkpoints/u-vit-cifar10/cifar10_uvit_small.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDIM-only public CIFAR-10 pretrained p0_high cross-check.")
    parser.add_argument("--models", nargs="+", default=["public_ddpm_unet", "public_uvit"], choices=sorted(MODEL_DEFAULTS))
    parser.add_argument("--checkpoint_public_ddpm_unet", default=MODEL_DEFAULTS["public_ddpm_unet"])
    parser.add_argument("--checkpoint_public_uvit", default=MODEL_DEFAULTS["public_uvit"])
    parser.add_argument("--output_dir", default="outputs/cifar_public_pretrained_ddim_crosscheck")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--split_seed", type=int, default=20260423)
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.1, 0.2, 0.35, 0.5])
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_real_stats", type=int, default=None)
    parser.add_argument("--num_probe", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--max_density_ratio", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


def _checkpoint_for_model(args: argparse.Namespace, model_name: str) -> str:
    return str(getattr(args, f"checkpoint_{model_name}"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _record_density_rows(rows: list[dict[str, object]], schedule, model_name: str, nfe: int, mode: str, strength: float, density: np.ndarray) -> None:
    edges = logsnr_bin_edges(schedule, len(density), torch.device("cpu")).double()
    centers = 0.5 * (edges[:-1] + edges[1:])
    t_centers = schedule.inverse_log_snr(centers.float()).double()
    for bin_index, value in enumerate(density):
        rows.append(
            {
                "architecture": model_name,
                "solver": "ddim",
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


def _record_gap_rows(rows: list[dict[str, object]], schedule, model_name: str, nfe: int, mode: str, strength: float, times: torch.Tensor) -> None:
    times_cpu = times.detach().float().cpu()
    lambdas = schedule.log_snr(times_cpu).detach().float().cpu()
    for step in range(times_cpu.numel() - 1):
        rows.append(
            {
                "architecture": model_name,
                "solver": "ddim",
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


@torch.no_grad()
def _generated_stats(model, schedule, extractor, noise_bank: torch.Tensor, labels: torch.Tensor, times: torch.Tensor, batch_size: int, device: torch.device):
    features = []
    total_nfe = 0
    for start in range(0, noise_bank.shape[0], batch_size):
        end = min(start + batch_size, noise_bank.shape[0])
        result = _sample_grid(
            model,
            schedule,
            noise_bank[start:end].to(device, non_blocking=True),
            labels[start:end].to(device, non_blocking=True),
            "ddim",
            times.numel() - 1,
            times,
        )
        total_nfe += result.nfe
        features.append(extractor(result.samples).detach().float().cpu().numpy())
    return feature_stats(np.concatenate(features, axis=0)[: noise_bank.shape[0]]), total_nfe


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    schedule = public_cifar_schedule()
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)

    _, val_loader = build_cifar10_loaders(_cifar_config(args, args.batch_size), download=False)
    _, probe_loader = build_cifar10_loaders(_cifar_config(args, args.probe_batch_size), download=False)
    num_real_stats = args.num_real_stats or args.num_samples
    real_stats_path = args.real_stats_cache or default_real_stats_path(
        args.data_root,
        split_seed=args.split_seed,
        num_samples=num_real_stats,
        feature_backend=args.feature_backend,
    )
    real_stats = load_or_compute_real_stats(
        real_stats_path,
        itertools.cycle(val_loader),
        extractor,
        device,
        num_real_stats,
        metadata={
            "dataset": "cifar10_val",
            "split_seed": args.split_seed,
            "feature_backend": args.feature_backend,
            "num_samples": num_real_stats,
        },
    )
    noise_path = args.noise_bank or default_noise_bank_path(args.data_root, args.seed, args.num_samples, (3, 32, 32))
    noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=(3, 32, 32), seed=args.seed)
    labels = torch.zeros(args.num_samples, dtype=torch.long)
    band_spec = radial_band_spec(32, 32, args.freq_bands, device)
    noise_id = noise_bank_id(noise_path)

    metrics_rows: list[dict[str, object]] = []
    p0_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    maps_payload: dict[str, np.ndarray] = {}

    for model_name in args.models:
        checkpoint = _checkpoint_for_model(args, model_name)
        model = build_public_cifar_model(model_name, checkpoint, device)
        difficulty = _difficulty_map(model, schedule, probe_loader, args.num_probe, args.time_bins, band_spec, device)
        difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
        maps_payload[f"{model_name}_difficulty"] = difficulty
        maps_payload[f"{model_name}_difficulty_norm"] = difficulty_norm

        for nfe in args.nfe:
            error_map = _solver_error_map(
                model,
                schedule,
                probe_loader,
                "ddim",
                nfe,
                args.num_probe,
                args.time_bins,
                band_spec,
                args.ref_substeps,
                device,
            )
            p0_high = (difficulty_norm * error_map)[:, _high_band_slice(args.freq_bands)].sum(axis=1)
            maps_payload[f"{model_name}_error_map_nfe_{nfe}"] = error_map
            maps_payload[f"{model_name}_p0_high_nfe_{nfe}"] = p0_high
            p0_rows.append(
                {
                    "architecture": model_name,
                    "solver": "ddim",
                    "nfe": nfe,
                    "difficulty_weighted_error": float((difficulty_norm * error_map).sum()),
                    "p0_high_sum": float(p0_high.sum()),
                    "total_error": float(error_map.sum()),
                    "num_probe": args.num_probe,
                    "ref_substeps": args.ref_substeps,
                    "time_bins": args.time_bins,
                    "freq_bands": args.freq_bands,
                    "argmax_bin": int(p0_high.argmax()),
                }
            )
            candidates: list[tuple[str, float, np.ndarray]] = [
                ("uniform", 0.0, np.ones(args.time_bins, dtype=np.float64))
            ]
            candidates.extend(
                ("p0_high", strength, _normalize_profile(p0_high, strength, args.max_density_ratio))
                for strength in args.strengths
            )
            baseline_fid = None
            for mode, strength, density in candidates:
                times = _time_grid_from_density(schedule, nfe, density, device)
                _record_density_rows(density_rows, schedule, model_name, nfe, mode, strength, density)
                _record_gap_rows(gap_rows, schedule, model_name, nfe, mode, strength, times)
                stats, total_nfe = _generated_stats(
                    model,
                    schedule,
                    extractor,
                    noise_bank,
                    labels,
                    times,
                    args.batch_size,
                    device,
                )
                fid = frechet_distance(real_stats, stats)
                if mode == "uniform":
                    baseline_fid = fid
                delta = fid - baseline_fid if baseline_fid is not None else 0.0
                metrics_rows.append(
                    {
                        "architecture": model_name,
                        "solver": "ddim",
                        "nfe": nfe,
                        "mode": mode,
                        "strength": strength,
                        "fid": fid,
                        "delta_vs_uniform": delta,
                        "num_samples": args.num_samples,
                        "num_real_stats": num_real_stats,
                        "num_probe": args.num_probe,
                        "ref_substeps": args.ref_substeps,
                        "time_bins": args.time_bins,
                        "freq_bands": args.freq_bands,
                        "max_density_ratio": args.max_density_ratio,
                        "density_min": float(density.min()),
                        "density_max": float(density.max()),
                        "density_argmax_bin": int(density.argmax()),
                        "total_model_calls": total_nfe,
                        "checkpoint": checkpoint,
                        "seed": args.seed,
                        "noise_bank_id": noise_id,
                        "noise_bank": str(noise_path),
                        "real_stats_cache": str(real_stats_path),
                        "feature_backend": args.feature_backend,
                    }
                )
                print(
                    f"{model_name} ddim nfe={nfe} {mode} strength={strength:g} "
                    f"fid={fid:.4f} delta={delta:.4f}",
                    flush=True,
                )
                _write_csv(output_dir / "ddim_crosscheck_metrics.csv", metrics_rows)
            _write_csv(output_dir / "ddim_crosscheck_p0_scores.csv", p0_rows)
            _write_csv(output_dir / "ddim_crosscheck_density.csv", density_rows)
            _write_csv(output_dir / "ddim_crosscheck_gaps.csv", gap_rows)
            np.savez(output_dir / "ddim_crosscheck_maps.npz", **maps_payload)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(output_dir / "ddim_crosscheck_metrics.csv", flush=True)
    print(output_dir / "ddim_crosscheck_p0_scores.csv", flush=True)
    print(output_dir / "ddim_crosscheck_density.csv", flush=True)
    print(output_dir / "ddim_crosscheck_gaps.csv", flush=True)


if __name__ == "__main__":
    main()
