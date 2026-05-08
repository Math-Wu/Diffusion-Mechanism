from __future__ import annotations

import torch

from dm.third_party.dit_official import DiT


class SiT(DiT):
    """Official-compatible SiT backbone for the public SiT-XL/2 checkpoint.

    The official SiT repository reuses the DiT transformer backbone and, for
    the ImageNet256 checkpoint, returns only the learned velocity channels when
    `learn_sigma=True`.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = super().forward(x, t, y)
        if self.learn_sigma:
            output, _sigma = output.chunk(2, dim=1)
        return output

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        guided, rest = model_out[:, :3], model_out[:, 3:]
        cond, uncond = torch.split(guided, len(guided) // 2, dim=0)
        half_guided = uncond + cfg_scale * (cond - uncond)
        guided = torch.cat([half_guided, half_guided], dim=0)
        return torch.cat([guided, rest], dim=1)


def SiT_XL_2(**kwargs) -> SiT:
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
