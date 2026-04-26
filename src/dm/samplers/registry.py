from __future__ import annotations

import torch

from dm.samplers.base import EpsModel, SamplerResult
from dm.samplers.ode import DDIMSampler, HeunSampler, MultistepODESampler
from dm.schedules import CosineVPSchedule


SAMPLERS = ("ddim", "heun", "deis", "dpmpp", "unipc")


def build_sampler(name: str, schedule: CosineVPSchedule):
    key = name.lower()
    if key == "ddim":
        return DDIMSampler(schedule)
    if key == "heun":
        return HeunSampler(schedule)
    if key == "deis":
        return MultistepODESampler(schedule, name="deis", max_order=3, damping=0.95)
    if key == "dpmpp":
        return MultistepODESampler(schedule, name="dpmpp", max_order=2, damping=1.0)
    if key == "unipc":
        return MultistepODESampler(schedule, name="unipc", max_order=3, damping=1.0)
    raise ValueError(f"Unknown sampler: {name}. Expected one of {SAMPLERS}")


@torch.no_grad()
def sample(
    model: EpsModel,
    schedule: CosineVPSchedule,
    x: torch.Tensor,
    solver: str,
    nfe: int,
) -> SamplerResult:
    sampler = build_sampler(solver, schedule)
    return sampler.sample(model, x, nfe)
