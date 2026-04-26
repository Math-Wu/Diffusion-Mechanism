from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dm.models.common import group_norm, timestep_embedding, zero_module


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.in_layers = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.time_layers = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.out_layers = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb = self.time_layers(time_emb)
        while emb.ndim < h.ndim:
            emb = emb[..., None]
        h = h + emb
        h = self.out_layers(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        head_dim = channels // self.num_heads
        q = q.reshape(batch * self.num_heads, head_dim, height * width).transpose(1, 2)
        k = k.reshape(batch * self.num_heads, head_dim, height * width)
        v = v.reshape(batch * self.num_heads, head_dim, height * width).transpose(1, 2)
        weight = torch.bmm(q, k) * (head_dim ** -0.5)
        weight = torch.softmax(weight, dim=-1)
        out = torch.bmm(weight, v).transpose(1, 2).reshape(batch, channels, height * width)
        out = self.proj(out).reshape(batch, channels, height, width)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class TimestepSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        base_channels: int,
        channel_mult: list[int],
        num_res_blocks: int,
        attention_resolutions: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.input_blocks = nn.ModuleList([TimestepSequential(nn.Conv2d(in_channels, base_channels, 3, padding=1))])
        input_block_channels = [base_channels]
        channels = base_channels
        downsample_factor = 1
        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [ResBlock(channels, out_channels, self.time_dim, dropout)]
                channels = out_channels
                if image_size // downsample_factor in attention_resolutions:
                    layers.append(AttentionBlock(channels))
                self.input_blocks.append(TimestepSequential(*layers))
                input_block_channels.append(channels)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepSequential(Downsample(channels)))
                input_block_channels.append(channels)
                downsample_factor *= 2

        self.middle_block = TimestepSequential(
            ResBlock(channels, channels, self.time_dim, dropout),
            AttentionBlock(channels),
            ResBlock(channels, channels, self.time_dim, dropout),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_channels = base_channels * mult
            for index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [ResBlock(channels + skip_channels, out_channels, self.time_dim, dropout)]
                channels = out_channels
                if image_size // downsample_factor in attention_resolutions:
                    layers.append(AttentionBlock(channels))
                if level and index == num_res_blocks:
                    layers.append(Upsample(channels))
                    downsample_factor //= 2
                self.output_blocks.append(TimestepSequential(*layers))

        self.out = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(channels, in_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t * 1000.0, self.time_embed[0].in_features)
        emb = self.time_embed(emb)
        hs: list[torch.Tensor] = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
