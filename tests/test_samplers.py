from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")

import torch

from dm.samplers import SAMPLERS, sample
from dm.schedules import CosineVPSchedule


class ZeroModel:
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


@pytest.mark.parametrize("solver", SAMPLERS)
@pytest.mark.parametrize("nfe", [4, 8, 15])
def test_sampler_nfe_accounting(solver, nfe):
    schedule = CosineVPSchedule()
    x = torch.randn(2, 3, 32, 32)
    result = sample(ZeroModel(), schedule, x, solver=solver, nfe=nfe)
    assert result.samples.shape == x.shape
    assert result.nfe == nfe
    assert torch.isfinite(result.samples).all()
