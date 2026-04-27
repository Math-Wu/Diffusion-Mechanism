from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")

import torch

from dm.analysis import fft_radial_band_energy, logsnr_bin_centers, logsnr_bin_index, radial_band_spec
from dm.schedules import CosineVPSchedule


def test_fft_radial_band_energy_conserves_l2_energy():
    x = torch.randn(4, 3, 32, 32)
    spec = radial_band_spec(32, 32, num_bands=8, device=x.device)
    band_energy = fft_radial_band_energy(x, spec).sum(dim=1)
    pixel_energy = x.square().sum(dim=(1, 2, 3))
    assert torch.allclose(band_energy, pixel_energy, rtol=1e-5, atol=1e-4)


def test_logsnr_bin_centers_are_valid_and_indexable():
    schedule = CosineVPSchedule()
    centers = logsnr_bin_centers(schedule, num_bins=16, device=torch.device("cpu"))
    assert centers.shape == (16,)
    assert torch.all(centers >= schedule.eps)
    assert torch.all(centers <= 1.0 - schedule.eps)
    indices = [logsnr_bin_index(schedule, t, 16) for t in centers]
    assert indices == list(range(16))
