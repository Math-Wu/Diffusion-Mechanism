from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")

import torch

from dm.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, model.state_dict(), 3, 12, {"run": {"name": "test"}})
    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 3
    assert ckpt["images_seen"] == 12
    assert "ema" in ckpt
