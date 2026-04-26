from __future__ import annotations

import argparse
import itertools

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.data import build_cifar10_loaders
from dm.experiment import build_schedule, load_model_from_checkpoint
from dm.metrics import build_feature_extractor, collect_features, feature_stats, frechet_distance
from dm.samplers import sample
from dm.utils import default_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def generated_batches(model, schedule, config, solver: str, nfe: int, batch_size: int, seed: int, total: int, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    remaining = total
    while remaining > 0:
        batch = min(batch_size, remaining)
        noise = torch.randn(batch, config["model"]["in_channels"], 32, 32, device=device, generator=generator)
        result = sample(model, schedule, noise, solver=solver, nfe=nfe)
        yield result.samples
        remaining -= batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    model, config, _ = load_model_from_checkpoint(args.config, args.checkpoint, device, use_ema=True)
    config["training"]["batch_size"] = args.batch_size
    _, val_loader = build_cifar10_loaders(config, download=False)
    schedule = build_schedule(config)
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    real = collect_features(itertools.cycle(val_loader), extractor, device, args.num_samples)
    fake = collect_features(
        generated_batches(model, schedule, config, args.solver, args.nfe, args.batch_size, args.seed, args.num_samples, device),
        extractor,
        device,
        args.num_samples,
    )
    fid = frechet_distance(feature_stats(real), feature_stats(fake))
    print(f"fid={fid:.6f} solver={args.solver} nfe={args.nfe} samples={args.num_samples}", flush=True)


if __name__ == "__main__":
    main()
