"""Minimal U-ViT official AutoencoderKL decoder adapter.

Adapted from `baofff/U-ViT/libs/autoencoder.py` for ImageNet256 latent
checkpoint decoding. This local copy keeps only the decoder path needed for
evaluation; encoder weights in the official checkpoint are intentionally
ignored.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        attn = torch.bmm(q, k) * (channels ** -0.5)
        attn = torch.nn.functional.softmax(attn, dim=2)
        v = v.reshape(batch, channels, height * width)
        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.reshape(batch, channels, height, width)
        return x + self.proj_out(h)


def make_attn(in_channels: int, attn_type: str = "vanilla") -> nn.Module:
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    if attn_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported autoencoder attention type: {attn_type}")


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: list[int] | tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        attn_type: str = "vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class FrozenAutoencoderKLDecoder(nn.Module):
    def __init__(self, pretrained_path: str, scale_factor: float = 0.18215):
        super().__init__()
        ddconfig = dict(
            double_z=True,
            z_channels=4,
            resolution=256,
            in_channels=3,
            out_ch=3,
            ch=128,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            dropout=0.0,
        )
        self.decoder = Decoder(**ddconfig)
        self.post_quant_conv = nn.Conv2d(4, ddconfig["z_channels"], 1)
        self.scale_factor = scale_factor
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, _unexpected = self.load_state_dict(state, strict=False)
        critical_missing = [key for key in missing if key.startswith(("decoder.", "post_quant_conv."))]
        if critical_missing:
            raise RuntimeError(f"U-ViT autoencoder decoder weights missing: {critical_missing[:10]}")
        self.eval()
        self.requires_grad_(False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.scale_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)


def get_decoder(pretrained_path: str, scale_factor: float = 0.18215) -> FrozenAutoencoderKLDecoder:
    return FrozenAutoencoderKLDecoder(pretrained_path, scale_factor=scale_factor)
