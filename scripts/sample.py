from __future__ import annotations

import argparse

import torch
from torchvision.utils import save_image

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.eval_utils import default_noise_bank_path, load_or_create_noise_bank, noise_bank_id, noise_batches
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
    parser.add_argument("--noise_bank")
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
    image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
    noise_shape = (int(config["model"]["in_channels"]), image_size, image_size)
    noise_path = args.noise_bank or default_noise_bank_path(config["data"].get("root", "data"), args.seed, args.num_samples, noise_shape)
    noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_samples, shape=noise_shape, seed=args.seed)
    total_nfe = 0
    for x in noise_batches(noise_bank, args.batch_size, device):
        result = sample(model, schedule, x, solver=args.solver, nfe=args.nfe)
        total_nfe += result.nfe
        all_samples.append(result.samples.detach().cpu())
    samples = torch.cat(all_samples, dim=0)
    torch.save(
        {
            "samples": samples,
            "solver": args.solver,
            "nfe": args.nfe,
            "seed": args.seed,
            "noise_bank": str(noise_path),
            "noise_bank_id": noise_bank_id(noise_path),
        },
        output_dir / "samples.pt",
    )
    save_image(((samples[:64] + 1.0) * 0.5).clamp(0.0, 1.0), output_dir / "grid.png", nrow=8)
    print(f"saved={output_dir} noise_bank_id={noise_bank_id(noise_path)} batches_nfe={total_nfe}", flush=True)


if __name__ == "__main__":
    main()
