from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.nn import functional as F
from torchvision import models


def _to_unit_range(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


class PixelFeatureExtractor(nn.Module):
    dim = 3 * 32 * 32

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _to_unit_range(images).flatten(1)


class InceptionFeatureExtractor(nn.Module):
    dim = 2048

    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights, aux_logits=True)
        self.model.fc = nn.Identity()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = _to_unit_range(images)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        x = (x - mean) / std
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


def build_feature_extractor(name: str, device: torch.device, allow_pixel_fallback: bool = False) -> nn.Module:
    key = name.lower()
    if key == "pixel":
        return PixelFeatureExtractor().to(device).eval()
    if key == "inception":
        try:
            return InceptionFeatureExtractor().to(device).eval()
        except Exception:
            if allow_pixel_fallback:
                return PixelFeatureExtractor().to(device).eval()
            raise
    raise ValueError(f"Unknown feature extractor: {name}")


@dataclass
class FeatureStats:
    mu: np.ndarray
    sigma: np.ndarray


@torch.no_grad()
def collect_features(
    batches,
    extractor: nn.Module,
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    count = 0
    for batch in batches:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        feat = extractor(images).detach().float().cpu().numpy()
        features.append(feat)
        count += feat.shape[0]
        if count >= max_samples:
            break
    if not features:
        raise ValueError("No features collected")
    return np.concatenate(features, axis=0)[:max_samples]


def feature_stats(features: np.ndarray) -> FeatureStats:
    return FeatureStats(mu=np.mean(features, axis=0), sigma=np.cov(features, rowvar=False))


def frechet_distance(real: FeatureStats, fake: FeatureStats) -> float:
    covmean, _ = linalg.sqrtm(real.sigma @ fake.sigma, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(real.sigma.shape[0]) * 1e-6
        covmean = linalg.sqrtm((real.sigma + offset) @ (fake.sigma + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = real.mu - fake.mu
    fid = diff.dot(diff) + np.trace(real.sigma + fake.sigma - 2.0 * covmean)
    if math.isnan(float(fid)):
        raise FloatingPointError("FID became NaN")
    return float(fid)
