from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dm.third_party.guided_diffusion.unet import UNetModel
from dm.third_party.uvit_official.uvit import UViT


class ImageNet64ValDataset(Dataset):
    """Downsampled ImageNet64 validation split stored in CIFAR-like NPZ format."""

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        payload = np.load(self.npz_path)
        self.data = payload["data"]
        self.labels = payload["labels"].astype(np.int64) - 1
        if self.data.ndim != 2 or self.data.shape[1] != 3 * 64 * 64:
            raise ValueError(f"Expected flattened ImageNet64 data with shape [N, 12288], got {self.data.shape}")
        if self.labels.min() < 0 or self.labels.max() >= 1000:
            raise ValueError("Expected ImageNet labels in 1..1000 before conversion to 0..999")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self.data[index]
        image = flat.reshape(3, 64, 64).astype(np.float32) / 255.0
        image = torch.from_numpy(image) * 2.0 - 1.0
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, label


def build_imagenet64_val_loader(
    npz_path: str | Path,
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    dataset = ImageNet64ValDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _continuous_to_999(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(0.0, 1.0) * 999.0).float()


class ADM64EpsWrapper(nn.Module):
    """OpenAI guided-diffusion ImageNet64 ADM wrapper returning epsilon only."""

    def __init__(self):
        super().__init__()
        self.model = UNetModel(
            image_size=64,
            in_channels=3,
            model_channels=192,
            out_channels=6,
            num_res_blocks=3,
            attention_resolutions=(2, 4, 8),
            dropout=0.1,
            channel_mult=(1, 2, 3, 4),
            num_classes=1000,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        timestep = _continuous_to_999(t)
        output = self.model(x, timestep, y=y)
        eps, _variance = torch.chunk(output, 2, dim=1)
        return eps


class UViT64EpsWrapper(nn.Module):
    """Official U-ViT ImageNet64 L/4 wrapper returning epsilon prediction."""

    def __init__(self):
        super().__init__()
        self.model = UViT(
            img_size=64,
            patch_size=4,
            in_chans=3,
            embed_dim=1024,
            depth=20,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=False,
            mlp_time_embed=False,
            num_classes=1000,
            use_checkpoint=False,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        timestep = _continuous_to_999(t)
        return self.model(x, timestep, y=y)


def load_imagenet64_pretrained_model(
    architecture: str,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    if architecture == "adm":
        model: nn.Module = ADM64EpsWrapper()
    elif architecture == "uvit":
        model = UViT64EpsWrapper()
    else:
        raise ValueError(f"Unknown ImageNet64 pretrained architecture: {architecture}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

