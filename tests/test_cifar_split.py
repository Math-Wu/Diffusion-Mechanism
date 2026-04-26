from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="torchvision not installed")

import numpy as np
from torchvision import datasets

from dm.data import get_cifar10_splits


def test_cifar_split_is_balanced_when_data_exists(tmp_path):
    try:
        dataset = datasets.CIFAR10(root=str(tmp_path), train=True, download=True)
    except Exception as exc:
        pytest.skip(f"CIFAR download unavailable: {exc}")
    train_idx, val_idx = get_cifar10_splits(tmp_path, download=False)
    assert len(train_idx) == 45_000
    assert len(val_idx) == 5_000
    counts = np.bincount(np.asarray(dataset.targets)[val_idx], minlength=10)
    assert counts.tolist() == [500] * 10
