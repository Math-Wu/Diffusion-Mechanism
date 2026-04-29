from __future__ import annotations

import argparse

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.imagenet64 import build_imagenet64_val_loader, load_imagenet64_pretrained_model
from dm.utils import default_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test ImageNet64 pretrained model adapters.")
    parser.add_argument("--data_npz", default="data/imagenet64/Imagenet64_val_npz/val_data.npz")
    parser.add_argument("--adm_checkpoint", default="checkpoints/guided-diffusion/64x64_diffusion.pt")
    parser.add_argument("--uvit_checkpoint", default="checkpoints/u-vit/imagenet64_uvit_large.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def smoke_one(name: str, checkpoint: str, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> None:
    model = load_imagenet64_pretrained_model(name, checkpoint, device)
    x = x.to(device)
    y = y.to(device)
    t = torch.linspace(0.05, 0.95, x.shape[0], device=device)
    eps = model(x, t, y)
    if eps.shape != x.shape:
        raise AssertionError(f"{name} output shape {tuple(eps.shape)} != input shape {tuple(x.shape)}")
    if not torch.isfinite(eps).all():
        raise AssertionError(f"{name} output contains non-finite values")
    print(
        f"smoke_ok model={name} batch={x.shape[0]} shape={tuple(eps.shape)} "
        f"label_min={int(y.min())} label_max={int(y.max())} "
        f"eps_mean={float(eps.mean()):.6f} eps_std={float(eps.std()):.6f}",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    set_seed(12345)
    device = default_device()
    loader = build_imagenet64_val_loader(args.data_npz, args.batch_size, num_workers=args.num_workers)
    x, y = next(iter(loader))
    if x.shape[1:] != (3, 64, 64):
        raise AssertionError(f"Unexpected ImageNet64 batch shape: {tuple(x.shape)}")
    if y.min() < 0 or y.max() >= 1000:
        raise AssertionError("Labels must be converted to 0..999")
    print(
        f"data_ok batch={x.shape[0]} shape={tuple(x.shape)} "
        f"x_range=({float(x.min()):.3f},{float(x.max()):.3f}) "
        f"label_range=({int(y.min())},{int(y.max())})",
        flush=True,
    )
    smoke_one("adm", args.adm_checkpoint, x, y, device)
    smoke_one("uvit", args.uvit_checkpoint, x, y, device)


if __name__ == "__main__":
    main()
