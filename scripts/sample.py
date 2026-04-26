from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.experiment import build_schedule, load_model_from_checkpoint
from dm.samplers import sample
from dm.utils import default_device, ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device()
    model, config, _ = load_model_from_checkpoint(args.config, args.checkpoint, device, use_ema=not args.raw_weights)
    schedule = build_schedule(config)
    output_dir = ensure_dir(args.output_dir)
    all_samples: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    remaining = args.num_samples
    total_nfe = 0
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        x = torch.randn(batch, config["model"]["in_channels"], 32, 32, device=device, generator=generator)
        result = sample(model, schedule, x, solver=args.solver, nfe=args.nfe)
        total_nfe += result.nfe
        all_samples.append(result.samples.detach().cpu())
        remaining -= batch
    samples = torch.cat(all_samples, dim=0)
    torch.save({"samples": samples, "solver": args.solver, "nfe": args.nfe}, output_dir / "samples.pt")
    save_image(((samples[:64] + 1.0) * 0.5).clamp(0.0, 1.0), output_dir / "grid.png", nrow=8)
    print(f"saved={output_dir} batches_nfe={total_nfe}", flush=True)


if __name__ == "__main__":
    main()
