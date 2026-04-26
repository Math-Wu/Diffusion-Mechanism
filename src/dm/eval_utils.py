from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from dm.metrics import FeatureStats, collect_features, feature_stats
from dm.utils import ensure_dir


def default_noise_bank_path(root: str | Path, seed: int, num_samples: int, shape: tuple[int, int, int]) -> Path:
    shape_tag = "x".join(str(value) for value in shape)
    return Path(root) / "noise_banks" / f"cifar10_seed{seed}_n{num_samples}_{shape_tag}.pt"


def noise_bank_id(path: str | Path) -> str:
    return Path(path).stem


def load_or_create_noise_bank(
    path: str | Path,
    *,
    num_samples: int,
    shape: tuple[int, int, int],
    seed: int,
) -> torch.Tensor:
    path = Path(path)
    expected_tail = tuple(shape)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        noise = payload["noise"] if isinstance(payload, dict) and "noise" in payload else payload
        if tuple(noise.shape[1:]) != expected_tail:
            raise ValueError(f"Noise bank {path} has shape {tuple(noise.shape)}, expected (*, {expected_tail})")
        if noise.shape[0] < num_samples:
            raise ValueError(f"Noise bank {path} has only {noise.shape[0]} samples, need {num_samples}")
        return noise[:num_samples].detach().float().cpu().contiguous()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((num_samples, *expected_tail), generator=generator, dtype=torch.float32)
    ensure_dir(path.parent)
    torch.save(
        {
            "noise": noise,
            "metadata": {
                "seed": seed,
                "num_samples": num_samples,
                "shape": expected_tail,
                "id": noise_bank_id(path),
            },
        },
        path,
    )
    return noise


def noise_batches(noise_bank: torch.Tensor, batch_size: int, device: torch.device) -> Iterable[torch.Tensor]:
    for start in range(0, noise_bank.shape[0], batch_size):
        yield noise_bank[start : start + batch_size].to(device, non_blocking=True)


def default_real_stats_path(
    root: str | Path,
    *,
    split_seed: int,
    num_samples: int,
    feature_backend: str,
) -> Path:
    return Path(root) / "fid_stats" / f"cifar10_val_{feature_backend}_split{split_seed}_n{num_samples}.npz"


def save_feature_stats(path: str | Path, stats: FeatureStats, metadata: dict | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez(
        path,
        mu=stats.mu,
        sigma=stats.sigma,
        metadata=json.dumps(metadata or {}, sort_keys=True),
    )


def load_feature_stats(path: str | Path) -> FeatureStats:
    data = np.load(path, allow_pickle=False)
    return FeatureStats(mu=data["mu"], sigma=data["sigma"])


def load_or_compute_real_stats(
    path: str | Path,
    batches,
    extractor: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    metadata: dict | None = None,
) -> FeatureStats:
    path = Path(path)
    if path.exists():
        return load_feature_stats(path)
    features = collect_features(batches, extractor, device, num_samples)
    stats = feature_stats(features)
    save_feature_stats(path, stats, metadata=metadata)
    return stats
