from __future__ import annotations

import torch

from dm.samplers.base import BaseSampler, CountedModel, EpsModel, SamplerResult


AB_COEFFS = {
    1: (1.0,),
    2: (1.5, -0.5),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


class DDIMSampler(BaseSampler):
    name = "ddim"

    @torch.no_grad()
    def sample(self, model: EpsModel, x: torch.Tensor, nfe: int) -> SamplerResult:
        counted = CountedModel(model)
        times = self.schedule.time_grid(nfe, x.device)
        batch = x.shape[0]
        for index in range(nfe):
            t = _batch_time(times[index], batch)
            t_next = _batch_time(times[index + 1], batch)
            eps = counted(x, t)
            x0 = self.schedule.eps_to_x0(x, t, eps)
            alpha_next, sigma_next = self.schedule.alpha_sigma(t_next)
            while alpha_next.ndim < x.ndim:
                alpha_next = alpha_next[..., None]
                sigma_next = sigma_next[..., None]
            x = alpha_next * x0 + sigma_next * eps
        return SamplerResult(samples=x, nfe=counted.nfe)


class HeunSampler(BaseSampler):
    name = "heun"

    @torch.no_grad()
    def sample(self, model: EpsModel, x: torch.Tensor, nfe: int) -> SamplerResult:
        counted = CountedModel(model)
        intervals = max(1, (nfe + 1) // 2)
        times = self.schedule.time_grid(intervals, x.device)
        batch = x.shape[0]
        remaining = nfe
        for index in range(intervals):
            t = _batch_time(times[index], batch)
            t_next = _batch_time(times[index + 1], batch)
            dt = (t_next - t).view(batch, *([1] * (x.ndim - 1)))
            eps = counted(x, t)
            drift = self.schedule.drift(x, t, eps)
            remaining -= 1
            if remaining > 0:
                x_euler = x + dt * drift
                eps_next = counted(x_euler, t_next)
                drift_next = self.schedule.drift(x_euler, t_next, eps_next)
                x = x + 0.5 * dt * (drift + drift_next)
                remaining -= 1
            else:
                x = x + dt * drift
        return SamplerResult(samples=x, nfe=counted.nfe)


class MultistepODESampler(BaseSampler):
    def __init__(self, schedule, name: str, max_order: int, damping: float = 1.0):
        super().__init__(schedule)
        self.name = name
        self.max_order = max_order
        self.damping = damping

    @torch.no_grad()
    def sample(self, model: EpsModel, x: torch.Tensor, nfe: int) -> SamplerResult:
        counted = CountedModel(model)
        times = self.schedule.time_grid(nfe, x.device)
        batch = x.shape[0]
        history: list[torch.Tensor] = []
        for index in range(nfe):
            t = _batch_time(times[index], batch)
            t_next = _batch_time(times[index + 1], batch)
            dt = (t_next - t).view(batch, *([1] * (x.ndim - 1)))
            eps = counted(x, t)
            drift = self.schedule.drift(x, t, eps)
            history.insert(0, drift)
            order = min(self.max_order, len(history))
            coeffs = AB_COEFFS[order]
            update = torch.zeros_like(x)
            for coeff, old_drift in zip(coeffs, history):
                update = update + coeff * old_drift
            x = x + self.damping * dt * update
            history = history[: self.max_order]
        return SamplerResult(samples=x, nfe=counted.nfe)
