from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _split_path(root: str | Path, split_seed: int) -> Path:
    return Path(root) / "splits" / f"cifar10_stratified_seed{split_seed}.npz"


def get_cifar10_splits(
    root: str | Path,
    train_size: int = 45_000,
    val_size: int = 5_000,
    split_seed: int = 20260423,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    split_file = _split_path(root, split_seed)
    if split_file.exists():
        data = np.load(split_file)
        return data["train_idx"], data["val_idx"]

    dataset = datasets.CIFAR10(root=str(root), train=True, download=download)
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(split_seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    val_per_class = val_size // 10
    for label in range(10):
        class_idx = np.flatnonzero(targets == label)
        rng.shuffle(class_idx)
        val_idx.extend(class_idx[:val_per_class].tolist())
        train_idx.extend(class_idx[val_per_class:].tolist())
    train_idx = np.asarray(train_idx[:train_size], dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_file, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


def build_cifar10_dataset(root: str | Path, train: bool, download: bool = False):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    return datasets.CIFAR10(root=str(root), train=train, download=download, transform=transform)


def build_cifar10_loaders(config: dict, download: bool = False) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    root = data_cfg.get("root", "data")
    train_idx, val_idx = get_cifar10_splits(
        root=root,
        train_size=int(data_cfg.get("train_size", 45_000)),
        val_size=int(data_cfg.get("val_size", 5_000)),
        split_seed=int(data_cfg.get("split_seed", 20260423)),
        download=download,
    )
    train_dataset = build_cifar10_dataset(root, train=True, download=False)
    val_dataset = build_cifar10_dataset(root, train=True, download=False)
    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(val_dataset, val_idx.tolist())
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def val_class_counts(root: str | Path, split_seed: int = 20260423) -> dict[int, int]:
    _, val_idx = get_cifar10_splits(root=root, split_seed=split_seed, download=False)
    dataset = datasets.CIFAR10(root=str(root), train=True, download=False)
    targets = np.asarray(dataset.targets)[val_idx]
    return {label: int((targets == label).sum()) for label in range(10)}
