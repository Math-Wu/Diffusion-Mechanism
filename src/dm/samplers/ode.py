from __future__ import annotations

import torch

from dm.samplers.base import BaseSampler, CountedModel, EpsModel, SamplerResult


AB_COEFFS = {
    1: (1.0,),
    2: (1.5, -0.5),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}


_DEIS_T_AB_CACHE: dict[tuple[float, float, int, int, float, int], tuple[torch.Tensor, torch.Tensor]] = {}


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


class DEISTABSampler(BaseSampler):
    """DEIS tAB sampler following qsh-zh/deis' exponential-integrator form."""

    name = "deis"

    def __init__(self, schedule, order: int = 3, ts_order: float = 2.0, integration_points: int = 10_000):
        super().__init__(schedule)
        self.order = int(order)
        self.ts_order = float(ts_order)
        self.integration_points = int(integration_points)

    def _official_time_grid(self, nfe: int) -> torch.Tensor:
        hi = float(1.0 - self.schedule.eps)
        lo = float(self.schedule.eps)
        base = torch.linspace(hi ** (1.0 / self.ts_order), lo ** (1.0 / self.ts_order), nfe + 1, dtype=torch.float64)
        return base.pow(self.ts_order).clamp(float(self.schedule.eps), float(1.0 - self.schedule.eps))

    def _coefficients(self, nfe: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            float(self.schedule.eps),
            float(getattr(self.schedule, "s", 0.008)),
            nfe,
            self.order,
            self.ts_order,
            self.integration_points,
        )
        if key in _DEIS_T_AB_CACHE:
            return _DEIS_T_AB_CACHE[key]

        times = self._official_time_grid(nfe)
        x_coef = torch.empty(nfe, dtype=torch.float64)
        eps_coef = torch.zeros(nfe, self.order + 1, dtype=torch.float64)
        alpha, _ = self.schedule.alpha_sigma(times)
        x_coef[:] = alpha[1:] / alpha[:-1].clamp_min(1e-12)

        for step in range(nfe):
            current_order = min(self.order, step)
            t_start = times[step]
            t_end = times[step + 1]
            ts_poly = times[step - current_order : step + 1]
            dt = (t_end - t_start) / self.integration_points
            t_inter = t_start + torch.arange(self.integration_points, dtype=torch.float64) * dt
            alpha_inter, sigma_inter = self.schedule.alpha_sigma(t_inter)
            alpha_end, _ = self.schedule.alpha_sigma(t_end.view(1))
            psi = alpha_end.squeeze(0) / alpha_inter.clamp_min(1e-12)
            integrand = psi * 0.5 * self.schedule.beta(t_inter) / sigma_inter.clamp_min(1e-12)

            for history_index in range(current_order + 1):
                coef_idx = current_order - history_index
                basis = torch.ones_like(t_inter)
                for basis_idx in range(current_order + 1):
                    if basis_idx == coef_idx:
                        continue
                    denominator = ts_poly[coef_idx] - ts_poly[basis_idx]
                    basis = basis * (t_inter - ts_poly[basis_idx]) / denominator
                eps_coef[step, history_index] = (integrand * basis).sum() * dt

        _DEIS_T_AB_CACHE[key] = (times.float(), torch.cat([x_coef[:, None], eps_coef], dim=1).float())
        return _DEIS_T_AB_CACHE[key]

    @torch.no_grad()
    def sample(self, model: EpsModel, x: torch.Tensor, nfe: int) -> SamplerResult:
        counted = CountedModel(model)
        times_cpu, coef_cpu = self._coefficients(nfe)
        times = times_cpu.to(device=x.device, dtype=x.dtype)
        coefs = coef_cpu.to(device=x.device, dtype=x.dtype)
        batch = x.shape[0]
        history: list[torch.Tensor] = []
        for index in range(nfe):
            t = _batch_time(times[index], batch)
            eps = counted(x, t)
            step_coef = coefs[index]
            x_coef = step_coef[0].view(*([1] * x.ndim))
            x_next = x_coef * x
            for coef, eps_pred in zip(step_coef[1:], [eps, *history]):
                x_next = x_next + coef.view(*([1] * x.ndim)) * eps_pred
            x = x_next
            history = [eps, *history[: self.order - 1]]
        return SamplerResult(samples=x, nfe=counted.nfe)
