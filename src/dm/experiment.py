from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from dm.checkpoint import load_checkpoint
from dm.config import load_config
from dm.models import build_model
from dm.schedules import CosineVPSchedule


def build_schedule(config: dict) -> CosineVPSchedule:
    diffusion_cfg = config.get("diffusion", {})
    if diffusion_cfg.get("schedule", "cosine") != "cosine":
        raise ValueError("Only cosine VP schedule is implemented")
    return CosineVPSchedule(eps=float(diffusion_cfg.get("eps", 1e-3)))


def diffusion_loss(
    model: torch.nn.Module,
    schedule: CosineVPSchedule,
    x0: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = x0.shape[0]
    device = x0.device
    eps_min = schedule.eps
    t = torch.rand(batch, device=device, generator=generator) * (1.0 - 2.0 * eps_min) + eps_min
    noise = torch.randn(x0.shape, device=device, generator=generator)
    x_t = schedule.q_sample(x0, t, noise)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    return loss, {"loss": float(loss.detach().cpu())}


def load_model_from_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[torch.nn.Module, dict, dict]:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(config).to(device)
    state_key = "ema" if use_ema and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model, config, checkpoint


def checkpoint_path_for_run(output_dir: str | Path, name: str = "last.pt") -> Path:
    return Path(output_dir) / "checkpoints" / name
