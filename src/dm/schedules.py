from __future__ import annotations

import math

import torch


class CosineVPSchedule:
    """Continuous cosine VP schedule with t=0 as data and t=1 as noise."""

    def __init__(self, eps: float = 1e-3, s: float = 0.008):
        self.eps = float(eps)
        self.s = float(s)
        self._theta0 = self._theta(torch.tensor(0.0)).item()
        self._norm = math.cos(self._theta0) ** 2

    def _theta(self, t: torch.Tensor) -> torch.Tensor:
        return (t + self.s) / (1.0 + self.s) * (math.pi / 2.0)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp(0.0, 1.0)
        value = torch.cos(self._theta(t)).pow(2) / self._norm
        return value.clamp(min=1e-12, max=1.0)

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alpha_bar(t)
        alpha = alpha_bar.sqrt()
        sigma = (1.0 - alpha_bar).clamp(min=1e-12).sqrt()
        return alpha, sigma

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        theta = self._theta(t).clamp(max=math.pi / 2.0 - 1e-5)
        return (math.pi / (1.0 + self.s)) * torch.tan(theta)

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        return 2.0 * (alpha.clamp_min(1e-12).log() - sigma.clamp_min(1e-12).log())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        while alpha.ndim < x0.ndim:
            alpha = alpha[..., None]
            sigma = sigma[..., None]
        return alpha * x0 + sigma * noise

    def eps_to_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        while alpha.ndim < x_t.ndim:
            alpha = alpha[..., None]
            sigma = sigma[..., None]
        return (x_t - sigma * eps) / alpha.clamp_min(1e-6)

    def drift(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        beta = self.beta(t)
        _, sigma = self.alpha_sigma(t)
        while beta.ndim < x_t.ndim:
            beta = beta[..., None]
            sigma = sigma[..., None]
        return -0.5 * beta * x_t + 0.5 * beta * eps / sigma.clamp_min(1e-6)

    def time_grid(self, steps: int, device: torch.device) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        t_hi = torch.tensor(1.0 - self.eps, device=device)
        t_lo = torch.tensor(self.eps, device=device)
        lambda_hi = self.log_snr(t_hi)
        lambda_lo = self.log_snr(t_lo)
        lambdas = torch.linspace(lambda_hi, lambda_lo, steps + 1, device=device)
        return self.inverse_log_snr(lambdas)

    def inverse_log_snr(self, log_snr: torch.Tensor) -> torch.Tensor:
        lo = torch.zeros_like(log_snr)
        hi = torch.ones_like(log_snr)
        for _ in range(64):
            mid = (lo + hi) * 0.5
            value = self.log_snr(mid)
            lo = torch.where(value > log_snr, mid, lo)
            hi = torch.where(value <= log_snr, mid, hi)
        return ((lo + hi) * 0.5).clamp(self.eps, 1.0 - self.eps)
