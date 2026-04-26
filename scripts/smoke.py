from __future__ import annotations

import argparse

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.config import load_config
from dm.experiment import build_schedule, diffusion_loss
from dm.models import build_model
from dm.samplers import SAMPLERS, sample
from dm.utils import default_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = default_device()
    for config_path in args.configs:
        config = load_config(config_path)
        model = build_model(config).to(device)
        schedule = build_schedule(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        loss, _ = diffusion_loss(model, schedule, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.eval()
        for solver in SAMPLERS:
            noise = torch.randn(args.batch_size, 3, 32, 32, device=device)
            result = sample(model, schedule, noise, solver=solver, nfe=4)
            assert result.samples.shape == noise.shape
            assert result.nfe == 4
            assert torch.isfinite(result.samples).all()
        print(f"smoke_ok {config_path}", flush=True)


if __name__ == "__main__":
    main()
