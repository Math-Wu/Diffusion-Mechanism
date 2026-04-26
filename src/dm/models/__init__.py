from __future__ import annotations

from dm.models.dit import DiT
from dm.models.unet import UNetModel
from dm.models.uvit import UViT


def build_model(config: dict):
    model_cfg = config["model"]
    architecture = model_cfg["architecture"].lower()
    if architecture == "unet":
        return UNetModel(**{k: v for k, v in model_cfg.items() if k != "architecture"})
    if architecture == "uvit":
        return UViT(**{k: v for k, v in model_cfg.items() if k != "architecture"})
    if architecture == "dit":
        return DiT(**{k: v for k, v in model_cfg.items() if k != "architecture"})
    raise ValueError(f"Unknown architecture: {architecture}")
