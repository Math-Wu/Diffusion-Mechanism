from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dm.schedules import LinearVPSchedule
from dm.third_party.dit_official import DiT_XL_2
from dm.third_party.guided_diffusion.unet import UNetModel
from dm.third_party.uvit_official.autoencoder import get_decoder as get_uvit_autoencoder_decoder
from dm.third_party.uvit_official.uvit import UViT


def _continuous_to_999(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(0.0, 1.0) * 999.0).float()


def _clean_state_dict(state):
    if isinstance(state, dict) and "ema" in state:
        state = state["ema"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _torch_load_checkpoint(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ADM256EpsWrapper(nn.Module):
    """OpenAI guided-diffusion ImageNet256 class-conditional ADM wrapper."""

    def __init__(self):
        super().__init__()
        self.model = UNetModel(
            image_size=256,
            in_channels=3,
            model_channels=256,
            out_channels=6,
            num_res_blocks=2,
            attention_resolutions=(8, 16, 32),
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            num_classes=1000,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = self.model(x, _continuous_to_999(t), y=y)
        eps, _variance = torch.chunk(output, 2, dim=1)
        return eps


class DiT256LatentEpsWrapper(nn.Module):
    """Official DiT-XL/2 ImageNet256 latent wrapper returning epsilon only."""

    def __init__(self):
        super().__init__()
        self.model = DiT_XL_2(input_size=32, in_channels=4, num_classes=1000, learn_sigma=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output = self.model(x, _continuous_to_999(t), y)
        eps, _variance = torch.chunk(output, 2, dim=1)
        return eps


class UViT256LatentEpsWrapper(nn.Module):
    """Official U-ViT-L/2 ImageNet256 latent wrapper returning epsilon."""

    def __init__(self):
        super().__init__()
        self.model = UViT(
            img_size=32,
            patch_size=2,
            in_chans=4,
            embed_dim=1024,
            depth=20,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=False,
            mlp_time_embed=False,
            num_classes=1001,
            use_checkpoint=False,
            conv=True,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x, _continuous_to_999(t), y=y)


def imagenet256_schedule() -> LinearVPSchedule:
    return LinearVPSchedule(beta_min=0.1, beta_max=20.0, eps=1e-4)


def load_imagenet256_model(name: str, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    if name == "adm256":
        model: nn.Module = ADM256EpsWrapper()
    elif name == "dit_xl_2":
        model = DiT256LatentEpsWrapper()
    elif name == "uvit_l_2":
        model = UViT256LatentEpsWrapper()
    else:
        raise ValueError(f"Unknown ImageNet256 model: {name}")
    state = _clean_state_dict(_torch_load_checkpoint(checkpoint_path))
    model.model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_autoencoder_kl(vae_dir: str | Path, device: torch.device) -> nn.Module:
    vae_dir = Path(vae_dir)
    if vae_dir.is_file():
        vae = get_uvit_autoencoder_decoder(str(vae_dir))
        vae.to(device)
        vae.eval()
        return vae
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise ImportError("diffusers is required to decode ImageNet256 latent checkpoints") from exc
    vae = AutoencoderKL.from_pretrained(str(vae_dir))
    vae.to(device)
    vae.eval()
    return vae


@torch.no_grad()
def decode_latents(vae: nn.Module, latents: torch.Tensor, scale_factor: float = 0.18215) -> torch.Tensor:
    decoded = vae.decode(latents / scale_factor)
    if isinstance(decoded, torch.Tensor):
        images = decoded
    elif hasattr(decoded, "sample"):
        images = decoded.sample
    else:
        images = decoded[0]
    return images.clamp(-1.0, 1.0)
