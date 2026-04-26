from __future__ import annotations

import torch
from torch import nn

from dm.models.common import MLP, modulate, timestep_embedding, zero_module


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp.unsqueeze(1) * h


class DiT(nn.Module):
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
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.head = zero_module(nn.Linear(hidden_size, patch_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        p = self.patch_size
        channels = self.in_channels
        h = w = self.grid_size
        x = patches.reshape(batch, h, w, channels, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(batch, channels, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        cond = self.time_embed(timestep_embedding(t * 1000.0, h.shape[-1]))
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        patches = self.head(h)
        return self.unpatchify(patches)
