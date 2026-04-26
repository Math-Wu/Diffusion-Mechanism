from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None,
    reason="torch/torchvision not installed",
)

import torch

from dm.eval_utils import default_noise_bank_path, load_or_create_noise_bank, noise_bank_id


def test_noise_bank_is_reused_and_sliceable(tmp_path):
    path = default_noise_bank_path(tmp_path, seed=123, num_samples=8, shape=(3, 4, 4))
    first = load_or_create_noise_bank(path, num_samples=8, shape=(3, 4, 4), seed=123)
    second = load_or_create_noise_bank(path, num_samples=4, shape=(3, 4, 4), seed=999)

    assert noise_bank_id(path) == "cifar10_seed123_n8_3x4x4"
    assert first.shape == (8, 3, 4, 4)
    assert second.shape == (4, 3, 4, 4)
    assert torch.equal(first[:4], second)
