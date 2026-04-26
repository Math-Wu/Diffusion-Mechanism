from __future__ import annotations

import argparse
import itertools

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.config import load_config
from dm.data import build_cifar10_loaders
from dm.eval_utils import (
    default_noise_bank_path,
    default_real_stats_path,
    load_or_compute_real_stats,
    load_or_create_noise_bank,
    noise_bank_id,
)
from dm.metrics import build_feature_extractor
from dm.utils import default_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--real_stats_cache")
    parser.add_argument("--feature_backend", default="inception", choices=["inception", "pixel"])
    parser.add_argument("--allow_pixel_fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config)
    config["training"]["batch_size"] = args.batch_size
    device = default_device()

    image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
    noise_shape = (int(config["model"]["in_channels"]), image_size, image_size)
    noise_path = args.noise_bank or default_noise_bank_path(config["data"].get("root", "data"), args.seed, args.num_samples, noise_shape)
    load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=noise_shape, seed=args.seed)

    _, val_loader = build_cifar10_loaders(config, download=False)
    extractor = build_feature_extractor(args.feature_backend, device, args.allow_pixel_fallback)
    real_stats_path = args.real_stats_cache or default_real_stats_path(
        config["data"].get("root", "data"),
        split_seed=int(config["data"].get("split_seed", 20260423)),
        num_samples=args.num_samples,
        feature_backend=args.feature_backend,
    )
    load_or_compute_real_stats(
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
    print(
        f"noise_bank={noise_path} noise_bank_id={noise_bank_id(noise_path)} real_stats_cache={real_stats_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
