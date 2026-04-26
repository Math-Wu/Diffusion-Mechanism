from __future__ import annotations

import argparse
import csv
import itertools
import time

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.data import build_cifar10_loaders
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
from dm.samplers import SAMPLERS, sample
from dm.utils import default_device, ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/result_a_cifar_medium")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 8, 10, 15, 20, 35, 50])
    parser.add_argument("--reference_solver", default="heun", choices=SAMPLERS)
    parser.add_argument("--reference_nfe", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--checkpoint_name", default="last.pt")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def generated_batches(model, schedule, solver: str, nfe: int, batch_size: int, noise_bank: torch.Tensor, device):
    total_nfe = 0
    for noise in noise_batches(noise_bank, batch_size, device):
        start = time.time()
        result = sample(model, schedule, noise, solver=solver, nfe=nfe)
        total_nfe += result.nfe
        yield result.samples, time.time() - start, result.nfe
    return total_nfe


@torch.no_grad()
def generated_feature_stats(model, schedule, extractor, solver, nfe, batch_size, noise_bank, total, device):
    features = []
    runtime = 0.0
    total_nfe = 0
    for samples, elapsed, calls in generated_batches(model, schedule, solver, nfe, batch_size, noise_bank, device):
        features.append(extractor(samples).detach().float().cpu().numpy())
        runtime += elapsed
        total_nfe += calls
    return feature_stats(__import__("numpy").concatenate(features, axis=0)[:total]), runtime, total_nfe


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.run_dirs):
        raise ValueError("--configs and --run_dirs must have the same length")
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    rows = []
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    real_stats_cache = None
    real_stats_path = None
    noise_bank = None
    noise_path = None
    noise_id = None
    for config_path, run_dir in zip(args.configs, args.run_dirs):
        ckpt = checkpoint_path_for_run(run_dir, name=args.checkpoint_name)
        model, config, checkpoint = load_model_from_checkpoint(config_path, ckpt, device, use_ema=True)
        config["training"]["batch_size"] = args.batch_size
        _, val_loader = build_cifar10_loaders(config, download=False)
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
            image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
            noise_shape = (int(config["model"]["in_channels"]), image_size, image_size)
            noise_path = args.noise_bank or default_noise_bank_path(
                config["data"].get("root", "data"), args.seed, args.num_samples, noise_shape
            )
            noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=noise_shape, seed=args.seed)
            noise_id = noise_bank_id(noise_path)
        schedule = build_schedule(config)
        architecture = config["model"]["architecture"]
        checkpoint_id = f"step_{int(checkpoint.get('step', -1))}_images_{int(checkpoint.get('images_seen', -1))}"
        ref_stats, ref_runtime, _ = generated_feature_stats(
            model, schedule, extractor, args.reference_solver, args.reference_nfe, args.batch_size, noise_bank, args.num_samples, device
        )
        ref_fid = frechet_distance(real_stats_cache, ref_stats)
        rows.append(
            {
                "architecture": architecture,
                "solver": f"{args.reference_solver}_ref",
                "nfe": args.reference_nfe,
                "fid": ref_fid,
                "delta_fid": 0.0,
                "wall_clock_sec": ref_runtime,
                "checkpoint": str(ckpt),
                "checkpoint_id": checkpoint_id,
                "seed": args.seed,
                "noise_bank_id": noise_id,
                "noise_bank": str(noise_path),
                "real_stats_cache": str(real_stats_path),
                "feature_backend": args.feature_backend,
                "num_samples": args.num_samples,
            }
        )
        for solver in SAMPLERS:
            for nfe in args.nfe:
                stats, runtime, _ = generated_feature_stats(
                    model, schedule, extractor, solver, nfe, args.batch_size, noise_bank, args.num_samples, device
                )
                fid = frechet_distance(real_stats_cache, stats)
                rows.append(
                    {
                        "architecture": architecture,
                        "solver": solver,
                        "nfe": nfe,
                        "fid": fid,
                        "delta_fid": fid - ref_fid,
                        "wall_clock_sec": runtime,
                        "checkpoint": str(ckpt),
                        "checkpoint_id": checkpoint_id,
                        "seed": args.seed,
                        "noise_bank_id": noise_id,
                        "noise_bank": str(noise_path),
                        "real_stats_cache": str(real_stats_path),
                        "feature_backend": args.feature_backend,
                        "num_samples": args.num_samples,
                    }
                )
                print(f"{architecture} {solver} nfe={nfe} fid={fid:.4f} delta={fid - ref_fid:.4f}", flush=True)
    output = output_dir / "metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output, flush=True)


if __name__ == "__main__":
    main()
