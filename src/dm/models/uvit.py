from __future__ import annotations

import torch
from torch import nn

from dm.models.common import MLP, timestep_embedding, zero_module


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class UViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth % 2 != 0:
            raise ValueError("UViT depth must be even for symmetric long skips")
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        half = depth // 2
        self.in_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(half)]
        )
        self.skip_projs = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size) for _ in range(half)])
        self.out_blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(half)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = zero_module(nn.Linear(hidden_size, patch_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        p = self.patch_size
        channels = self.in_channels
        h = w = self.grid_size
        x = patches.reshape(batch, h, w, channels, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(batch, channels, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        time_token = self.time_embed(timestep_embedding(t * 1000.0, tokens.shape[-1])).unsqueeze(1)
        h = torch.cat([time_token, tokens], dim=1) + self.pos_embed
        skips: list[torch.Tensor] = []
        for block in self.in_blocks:
            h = block(h)
            skips.append(h)
        for block, proj in zip(self.out_blocks, self.skip_projs):
            skip = skips.pop()
            h = proj(torch.cat([h, skip], dim=-1))
            h = block(h)
        h = self.norm(h[:, 1:])
        patches = self.head(h)
        return self.unpatchify(patches)
