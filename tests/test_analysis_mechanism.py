from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")

import torch

from dm.analysis import (
    fft_radial_band_energy,
    high_band_slice,
    logsnr_bin_centers,
    logsnr_bin_index,
    radial_band_spec,
    radial_power_profile,
)
from dm.schedules import CosineVPSchedule


def test_fft_radial_band_energy_conserves_l2_energy():
    x = torch.randn(4, 3, 32, 32)
    spec = radial_band_spec(32, 32, num_bands=8, device=x.device)
    band_energy = fft_radial_band_energy(x, spec).sum(dim=1)
    pixel_energy = x.square().sum(dim=(1, 2, 3))
    assert torch.allclose(band_energy, pixel_energy, rtol=1e-5, atol=1e-4)


def test_radial_power_profile_matches_l2_energy():
    x = torch.randn(3, 3, 16, 16)
    profile = radial_power_profile(x, num_bins=6)
    assert profile.energy.shape == (3, 6)
    assert profile.mean_power.shape == (3, 6)
    assert torch.allclose(profile.energy.sum(dim=1), x.square().sum(dim=(1, 2, 3)), rtol=1e-5, atol=1e-4)
    assert high_band_slice(4) == slice(3, 4)


def test_logsnr_bin_centers_are_valid_and_indexable():
    schedule = CosineVPSchedule()
    centers = logsnr_bin_centers(schedule, num_bins=16, device=torch.device("cpu"))
    assert centers.shape == (16,)
    assert torch.all(centers >= schedule.eps)
    assert torch.all(centers <= 1.0 - schedule.eps)
    indices = [logsnr_bin_index(schedule, t, 16) for t in centers]
    assert indices == list(range(16))
