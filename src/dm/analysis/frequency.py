from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RadialPowerProfile:
    """Radial Fourier power profile for image batches."""

    energy: torch.Tensor
    mean_power: torch.Tensor
    counts: torch.Tensor
    edges: torch.Tensor


def radial_frequency_edges(height: int, width: int, num_bins: int, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return torch.linspace(0.0, float(radius.max()) + 1e-8, num_bins + 1, device=device)


def radial_power_profile(x: torch.Tensor, num_bins: int) -> RadialPowerProfile:
    """Return per-sample radial FFT energy and per-coefficient mean power."""
    if x.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W], got shape {tuple(x.shape)}")
    _, channels, height, width = x.shape
    device = x.device
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    edges = radial_frequency_edges(height, width, num_bins, device)
    spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    energies = []
    counts = []
    for index in range(num_bins):
        if index == num_bins - 1:
            mask = (radius >= edges[index]) & (radius <= edges[index + 1])
        else:
            mask = (radius >= edges[index]) & (radius < edges[index + 1])
        mask_f = mask.float()
        energies.append(torch.einsum("bchw,hw->b", power, mask_f))
        counts.append(mask_f.sum() * channels)
    energy = torch.stack(energies, dim=1)
    count_tensor = torch.stack(counts).clamp_min(1.0)
    return RadialPowerProfile(
        energy=energy,
        mean_power=energy / count_tensor[None, :],
        counts=count_tensor,
        edges=edges,
    )


def frequency_band_names(num_bands: int) -> list[str]:
    if num_bands == 4:
        return ["low", "mid_low", "mid_high", "high"]
    return [f"band_{index}" for index in range(num_bands)]


def high_band_slice(num_bands: int) -> slice:
    return slice(max(0, int(torch.ceil(torch.tensor(num_bands * 0.625)).item())), num_bands)


def low_mid_band_slice(num_bands: int) -> slice:
    return slice(0, max(1, int(torch.ceil(torch.tensor(num_bands * 0.625)).item())))


def safe_normalize(values: torch.Tensor, dim: int | tuple[int, ...] | None = None, eps: float = 1e-12) -> torch.Tensor:
    if dim is None:
        return values / values.sum().clamp_min(eps)
    return values / values.sum(dim=dim, keepdim=True).clamp_min(eps)
