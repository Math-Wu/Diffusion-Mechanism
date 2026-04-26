from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")

import torch

from dm.schedules import CosineVPSchedule


def test_cosine_schedule_ranges_and_monotonicity():
    schedule = CosineVPSchedule()
    t = torch.linspace(schedule.eps, 1.0 - schedule.eps, 128)
    alpha, sigma = schedule.alpha_sigma(t)
    log_snr = schedule.log_snr(t)
    assert torch.all(alpha > 0)
    assert torch.all(sigma > 0)
    assert torch.all(log_snr[:-1] > log_snr[1:])


def test_time_grid_descends_from_noise_to_data():
    schedule = CosineVPSchedule()
    grid = schedule.time_grid(8, torch.device("cpu"))
    assert grid.shape == (9,)
    assert torch.all(grid[:-1] > grid[1:])
