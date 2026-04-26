from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.config import load_config
from dm.models import build_model
from dm.utils import count_parameters, default_device, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/budget_reports/cifar_medium.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def measure_ms(model: torch.nn.Module, image_size: int, batch_size: int, iters: int, device: torch.device) -> float:
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    t = torch.rand(batch_size, device=device)
    for _ in range(3):
        model(x, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        model(x, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - start) * 1000.0 / iters


def main() -> None:
    args = parse_args()
    device = default_device()
    rows = []
    for config_path in args.configs:
        config = load_config(config_path)
        model = build_model(config).to(device).eval()
        rows.append(
            {
                "config": config_path,
                "architecture": config["model"]["architecture"],
                "parameters": count_parameters(model),
                "forward_ms_batch": f"{measure_ms(model, config['model']['image_size'], args.batch_size, args.iters, device):.4f}",
                "batch_size": args.batch_size,
            }
        )
    output = ensure_dir(Path(args.output).parent) / Path(args.output).name
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output, flush=True)


if __name__ == "__main__":
    main()
