from __future__ import annotations

from dataclasses import dataclass

import torch

from dm.schedules import CosineVPSchedule


@dataclass(frozen=True)
class BandSpec:
    masks: torch.Tensor
    edges: torch.Tensor


def radial_band_spec(height: int, width: int, num_bands: int, device: torch.device) -> BandSpec:
    """Build radial FFT masks for orthonormal 2D spectra."""
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    edges = torch.linspace(0.0, float(radius.max()) + 1e-8, num_bands + 1, device=device)
    masks = []
    for index in range(num_bands):
        if index == num_bands - 1:
            mask = (radius >= edges[index]) & (radius <= edges[index + 1])
        else:
            mask = (radius >= edges[index]) & (radius < edges[index + 1])
        masks.append(mask.float())
    return BandSpec(masks=torch.stack(masks, dim=0), edges=edges)


def fft_radial_band_energy(x: torch.Tensor, band_spec: BandSpec) -> torch.Tensor:
    """Return per-sample L2 spectral energy in each radial band."""
    spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    return torch.einsum("bchw,rhw->br", power, band_spec.masks)


def logsnr_bin_edges(schedule: CosineVPSchedule, num_bins: int, device: torch.device) -> torch.Tensor:
    t_hi = torch.tensor(1.0 - schedule.eps, device=device)
    t_lo = torch.tensor(schedule.eps, device=device)
    lambda_hi = schedule.log_snr(t_hi)
    lambda_lo = schedule.log_snr(t_lo)
    return torch.linspace(lambda_hi, lambda_lo, num_bins + 1, device=device)


def logsnr_bin_centers(schedule: CosineVPSchedule, num_bins: int, device: torch.device) -> torch.Tensor:
    edges = logsnr_bin_edges(schedule, num_bins, device)
    return schedule.inverse_log_snr(0.5 * (edges[:-1] + edges[1:]))


def logsnr_bin_index(schedule: CosineVPSchedule, t: torch.Tensor, num_bins: int) -> int:
    edges = logsnr_bin_edges(schedule, num_bins, t.device)
    value = schedule.log_snr(t)
    index = torch.searchsorted(edges, value.clamp(edges[0], edges[-1]), right=True).item() - 1
    return max(0, min(num_bins - 1, int(index)))
