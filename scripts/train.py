from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.checkpoint import save_checkpoint
from dm.config import load_config, save_config
from dm.data import build_cifar10_loaders
from dm.ema import EMA
from dm.experiment import build_schedule, diffusion_loss
from dm.models import build_model
from dm.utils import cycle, default_device, ensure_dir, images_to_steps, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config["run"]["name"] = args.run_name
    if args.output_dir:
        config["run"]["output_dir"] = args.output_dir
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.debug:
        config["training"]["images_seen"] = int(config["training"]["batch_size"]) * 5
        config["training"]["checkpoint_images"] = int(config["training"]["batch_size"]) * 5
        config["data"]["num_workers"] = 0
    if args.max_steps:
        config["training"]["images_seen"] = int(config["training"]["batch_size"]) * args.max_steps

    set_seed(int(config["run"]["seed"]))
    device = default_device()
    output_dir = ensure_dir(config["run"]["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    save_config(config, output_dir / "config.yaml")
    shutil.copy2(args.config, output_dir / "source_config.yaml")

    train_loader, _ = build_cifar10_loaders(config, download=args.download)
    train_iter = cycle(train_loader)
    model = build_model(config).to(device)
    schedule = build_schedule(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.999),
    )
    ema = EMA(model, decay=float(config["training"]["ema_decay"]))
    batch_size = int(config["training"]["batch_size"])
    total_steps = images_to_steps(int(config["training"]["images_seen"]), batch_size)
    checkpoint_interval = max(1, images_to_steps(int(config["training"]["checkpoint_images"]), batch_size))
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    amp = config["training"].get("amp", "none")
    use_amp = device.type == "cuda" and amp in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == "fp16")
    start = time.time()

    model.train()
    for step in range(1, total_steps + 1):
        images, _ = next(train_iter)
        images = images.to(device, non_blocking=True)
        if warmup_steps and step <= warmup_steps:
            scale = step / warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = float(config["training"]["lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss, metrics = diffusion_loss(model, schedule, images)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        if step % int(config["training"].get("log_interval", 100)) == 0 or step == 1:
            elapsed = max(1e-6, time.time() - start)
            images_seen = step * batch_size
            print(
                f"step={step}/{total_steps} images={images_seen} "
                f"loss={metrics['loss']:.6f} imgs_per_sec={images_seen / elapsed:.2f}",
                flush=True,
            )
        if step % checkpoint_interval == 0 or step == total_steps:
            images_seen = step * batch_size
            save_checkpoint(
                checkpoints_dir / f"step_{step:08d}.pt",
                model,
                optimizer,
                ema.state_dict(),
                step,
                images_seen,
                config,
            )
            save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, ema.state_dict(), step, images_seen, config)
    print("training_complete", flush=True)


if __name__ == "__main__":
    main()
