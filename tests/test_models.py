from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")

import torch

from dm.config import load_config
from dm.models import build_model


@pytest.mark.parametrize("config_path", sorted(["configs/cifar_medium/unet.yaml", "configs/cifar_medium/uvit.yaml", "configs/cifar_medium/dit.yaml"]))
def test_model_forward_shape(config_path):
    config = load_config(config_path)
    model = build_model(config)
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = model(x, t)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
