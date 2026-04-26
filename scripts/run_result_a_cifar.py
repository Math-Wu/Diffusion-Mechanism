from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.data import build_cifar10_loaders
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.metrics import build_feature_extractor, collect_features, feature_stats, frechet_distance
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
    parser.add_argument("--reference_nfe", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def generated_batches(model, schedule, config, solver: str, nfe: int, batch_size: int, seed: int, total: int, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    remaining = total
    total_nfe = 0
    while remaining > 0:
        batch = min(batch_size, remaining)
        noise = torch.randn(batch, config["model"]["in_channels"], 32, 32, device=device, generator=generator)
        start = time.time()
        result = sample(model, schedule, noise, solver=solver, nfe=nfe)
        total_nfe += result.nfe
        yield result.samples, time.time() - start, result.nfe
        remaining -= batch
    return total_nfe


@torch.no_grad()
def generated_feature_stats(model, schedule, config, extractor, solver, nfe, batch_size, seed, total, device):
    features = []
    runtime = 0.0
    total_nfe = 0
    for samples, elapsed, calls in generated_batches(model, schedule, config, solver, nfe, batch_size, seed, total, device):
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
    for config_path, run_dir in zip(args.configs, args.run_dirs):
        ckpt = checkpoint_path_for_run(run_dir)
        model, config, checkpoint = load_model_from_checkpoint(config_path, ckpt, device, use_ema=True)
        config["training"]["batch_size"] = args.batch_size
        _, val_loader = build_cifar10_loaders(config, download=False)
        if real_stats_cache is None:
            real = collect_features(itertools.cycle(val_loader), extractor, device, args.num_samples)
            real_stats_cache = feature_stats(real)
        schedule = build_schedule(config)
        architecture = config["model"]["architecture"]
        ref_stats, ref_runtime, _ = generated_feature_stats(
            model, schedule, config, extractor, "heun", args.reference_nfe, args.batch_size, args.seed, args.num_samples, device
        )
        ref_fid = frechet_distance(real_stats_cache, ref_stats)
        rows.append(
            {
                "architecture": architecture,
                "solver": "heun_ref",
                "nfe": args.reference_nfe,
                "fid": ref_fid,
                "delta_fid": 0.0,
                "runtime_sec": ref_runtime,
                "checkpoint": str(ckpt),
                "seed": args.seed,
            }
        )
        for solver in SAMPLERS:
            for nfe in args.nfe:
                stats, runtime, _ = generated_feature_stats(
                    model, schedule, config, extractor, solver, nfe, args.batch_size, args.seed, args.num_samples, device
                )
                fid = frechet_distance(real_stats_cache, stats)
                rows.append(
                    {
                        "architecture": architecture,
                        "solver": solver,
                        "nfe": nfe,
                        "fid": fid,
                        "delta_fid": fid - ref_fid,
                        "runtime_sec": runtime,
                        "checkpoint": str(ckpt),
                        "seed": args.seed,
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
