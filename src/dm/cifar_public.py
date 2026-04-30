from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dm.schedules import LinearVPSchedule
from dm.third_party.uvit_official.uvit import UViT


class DiffusersDDPMCifarWrapper(nn.Module):
    """Wrap a diffusers CIFAR-10 UNet2DModel as epsilon(x_t, t)."""

    def __init__(self, checkpoint_dir: str | Path):
        super().__init__()
        try:
            from diffusers import UNet2DModel
        except ImportError as exc:
            raise ImportError("diffusers is required for checkpoints/ddpm-cifar10-32") from exc
        self.model = UNet2DModel.from_pretrained(str(checkpoint_dir))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        del y
        return self.model(x, t * 999.0).sample


class OfficialUViTCifarWrapper(nn.Module):
    """Official U-ViT CIFAR-10 S/2 checkpoint wrapper as epsilon(x_t, t)."""

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        self.model = UViT(
            img_size=32,
            patch_size=2,
            in_chans=3,
            embed_dim=512,
            depth=12,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=False,
            mlp_time_embed=False,
            num_classes=-1,
            conv=True,
            skip=True,
        )
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        del y
        return self.model(x, t * 999.0)


def build_public_cifar_model(name: str, checkpoint: str | Path, device: torch.device) -> nn.Module:
    key = name.lower()
    if key in {"ddpm_unet", "google_ddpm_unet", "public_ddpm_unet"}:
        model = DiffusersDDPMCifarWrapper(checkpoint)
    elif key in {"uvit", "official_uvit", "public_uvit"}:
        model = OfficialUViTCifarWrapper(checkpoint)
    else:
        raise ValueError(f"Unknown public CIFAR model: {name}")
    return model.to(device).eval()


def public_cifar_schedule() -> LinearVPSchedule:
    return LinearVPSchedule(beta_min=0.1, beta_max=20.0, eps=1e-4)
