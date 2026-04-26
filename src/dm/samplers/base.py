from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from dm.schedules import CosineVPSchedule


class EpsModel(Protocol):
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class SamplerResult:
    samples: torch.Tensor
    nfe: int


class CountedModel:
    def __init__(self, model: EpsModel):
        self.model = model
        self.nfe = 0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.model(x, t)


class BaseSampler:
    name = "base"

    def __init__(self, schedule: CosineVPSchedule):
        self.schedule = schedule

    @torch.no_grad()
    def sample(self, model: EpsModel, x: torch.Tensor, nfe: int) -> SamplerResult:
        raise NotImplementedError
