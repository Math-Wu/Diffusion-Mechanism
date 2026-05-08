from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dm.imagenet256 import _clean_state_dict, _torch_load_checkpoint, load_autoencoder_kl
from dm.third_party.sit_official import SiT_XL_2


MODEL_SHAPES = {"sit_xl_2": (4, 32, 32)}


class SiT256VelocityWrapper(nn.Module):
    """SiT-XL/2 ImageNet256 latent flow wrapper returning velocity."""

    def __init__(self):
        super().__init__()
        self.model = SiT_XL_2(input_size=32, in_channels=4, num_classes=1000, learn_sigma=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x, t, y)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        return self.model.forward_with_cfg(x, t, y, cfg_scale)


def load_sit_xl_2(checkpoint_path: str | Path, device: torch.device) -> SiT256VelocityWrapper:
    model = SiT256VelocityWrapper()
    state = _clean_state_dict(_torch_load_checkpoint(checkpoint_path))
    model.model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


__all__ = ["MODEL_SHAPES", "SiT256VelocityWrapper", "load_autoencoder_kl", "load_sit_xl_2"]
