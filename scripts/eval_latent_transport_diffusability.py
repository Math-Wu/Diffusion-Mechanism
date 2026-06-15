from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dm_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()


torch = None
fft_radial_band_energy = None
radial_band_spec = None
load_autoencoder_kl = None
imagenet256_schedule = None
load_imagenet256_model = None
load_or_create_noise_bank = None
sample_grid = None
_default_noise_bank_path = None
default_device = None
set_seed = None


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _load_torch_deps() -> None:
    global torch
    global fft_radial_band_energy
    global radial_band_spec
    global load_autoencoder_kl
    global imagenet256_schedule
    global load_imagenet256_model
    global load_or_create_noise_bank
    global sample_grid
    global _default_noise_bank_path
    global default_device
    global set_seed
    if torch is not None:
        return
    import torch as torch_module

    from dm.analysis import fft_radial_band_energy as fft_radial_band_energy_fn
    from dm.analysis import radial_band_spec as radial_band_spec_fn
    from dm.eval_utils import load_or_create_noise_bank as load_or_create_noise_bank_fn
    from dm.imagenet256 import imagenet256_schedule as imagenet256_schedule_fn
    from dm.imagenet256 import load_autoencoder_kl as load_autoencoder_kl_fn
    from dm.imagenet256 import load_imagenet256_model as load_imagenet256_model_fn
    from dm.utils import default_device as default_device_fn
    from dm.utils import set_seed as set_seed_fn
    from run_imagenet256_pretrained_sweep import _default_noise_bank_path as _default_noise_bank_path_fn
    from run_imagenet256_pretrained_sweep import sample_grid as sample_grid_fn

    torch = torch_module
    fft_radial_band_energy = fft_radial_band_energy_fn
    radial_band_spec = radial_band_spec_fn
    load_autoencoder_kl = load_autoencoder_kl_fn
    imagenet256_schedule = imagenet256_schedule_fn
    load_imagenet256_model = load_imagenet256_model_fn
    load_or_create_noise_bank = load_or_create_noise_bank_fn
    sample_grid = sample_grid_fn
    _default_noise_bank_path = _default_noise_bank_path_fn
    default_device = default_device_fn
    set_seed = set_seed_fn


LATENT_MODELS = ("dit_xl_2", "uvit_l_2")
MODEL_LABELS = {"dit_xl_2": "DiT-XL/2", "uvit_l_2": "U-ViT-L/2"}
SOLVER_LABELS = {"ddim": "DDIM", "heun": "Heun", "dpmpp": "DPM++", "unipc": "UniPC"}
SOLVER_COLORS = {"ddim": "#4C78A8", "heun": "#F58518", "dpmpp": "#54A24B", "unipc": "#B279A2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 Latent-Transport Diffusability diagnostic. It estimates decoder spectral gain "
            "G_{latent band -> image band}, then tests whether decoder-weighted latent P2 explains "
            "ImageNet256 low-NFE quality better than raw latent P2."
        )
    )
    parser.add_argument("--output_dir", default="outputs/latent_transport_diffusability_imagenet256")
    parser.add_argument("--models", nargs="+", default=list(LATENT_MODELS), choices=LATENT_MODELS)
    parser.add_argument("--dit_vae_dir", default="checkpoints/DiT-XL:2")
    parser.add_argument("--uvit_vae_dir", default="checkpoints/U-ViT-L:2/autoencoder_kl.pth")
    parser.add_argument("--dit_checkpoint", default="checkpoints/DiT-XL:2/DiT-XL-2-256x256.pt")
    parser.add_argument("--uvit_checkpoint", default="checkpoints/U-ViT-L:2/imagenet256_uvit_large.pth")
    parser.add_argument("--latent_p2_root", default="outputs/imagenet256_p2_remote")
    parser.add_argument(
        "--decoded_p2_scores",
        default=(
            "outputs/imagenet256_decoded_x0_p2_remote/combined_analysis/"
            "imagenet256_decoded_x0_p2_combined_scores.csv"
        ),
    )
    parser.add_argument(
        "--latent_p2_scores",
        default="outputs/imagenet256_p2_remote/combined_analysis/imagenet256_p2_combined_scores.csv",
    )
    parser.add_argument("--num_anchors", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--latent_bands", type=int, default=8)
    parser.add_argument("--image_bands", type=int, default=8)
    parser.add_argument("--probes_per_band", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--entangle_threshold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--anchor_source", choices=["gaussian", "reference_x0"], default="gaussian")
    parser.add_argument("--anchor_reference_nfe", type=int, default=64)
    parser.add_argument("--anchor_noise_bank")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_gain", action="store_true", help="Only recompute joined scores/figures from existing gain CSVs.")
    parser.add_argument("--skip_analysis", action="store_true", help="Only estimate decoder spectral gain.")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _vae_dir_for_model(args: argparse.Namespace, model: str) -> str:
    if model == "dit_xl_2":
        return args.dit_vae_dir
    if model == "uvit_l_2":
        return args.uvit_vae_dir
    raise ValueError(f"Unsupported latent model: {model}")


def _checkpoint_for_model(args: argparse.Namespace, model: str) -> str:
    if model == "dit_xl_2":
        return args.dit_checkpoint
    if model == "uvit_l_2":
        return args.uvit_checkpoint
    raise ValueError(f"Unsupported latent model: {model}")


def _decode_unclamped(vae, latents: torch.Tensor, scale_factor: float = 0.18215):
    if vae.__class__.__name__ == "FrozenAutoencoderKLDecoder":
        decoded = vae.decode(latents)
    else:
        decoded = vae.decode(latents / scale_factor)
    if isinstance(decoded, torch.Tensor):
        return decoded
    if hasattr(decoded, "sample"):
        return decoded.sample
    return decoded[0]


def _rms_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = x.flatten(1).square().mean(dim=1).sqrt().clamp_min(eps)
    return x / denom.view(x.shape[0], *([1] * (x.ndim - 1)))


def _radial_project(x: torch.Tensor, band_spec, band_index: int) -> torch.Tensor:
    spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    projected = torch.fft.ifft2(spectrum * band_spec.masks[band_index][None, None], dim=(-2, -1), norm="ortho")
    return projected.real


def _band_names(num_bands: int) -> list[str]:
    return [f"band_{index}" for index in range(num_bands)]


def _high_band_start(num_bands: int) -> int:
    return max(0, int(math.ceil(num_bands * 0.625)))


def _anchor_noise_path(model: str, seed: int, num_anchors: int, anchor_noise_bank: str | None) -> Path:
    if anchor_noise_bank:
        path = Path(anchor_noise_bank)
        if len(LATENT_MODELS) > 1 and path.is_dir():
            return path / f"ltd_{model}_reference_seed{seed}_n{num_anchors}_4x32x32.pt"
        return path
    return Path("data/noise_banks") / f"ltd_{model}_reference_seed{seed}_n{num_anchors}_4x32x32.pt"


def _reference_x0_anchors(
    *,
    model_name: str,
    checkpoint: str,
    num_anchors: int,
    batch_size: int,
    reference_nfe: int,
    seed: int,
    anchor_noise_bank: str | None,
    device,
) -> torch.Tensor:
    schedule = imagenet256_schedule()
    model = load_imagenet256_model(model_name, checkpoint, device)
    noise_path = _anchor_noise_path(model_name, seed + 990_001, num_anchors, anchor_noise_bank)
    noise_bank = load_or_create_noise_bank(
        noise_path,
        num_samples=num_anchors,
        shape=(4, 32, 32),
        seed=seed + 990_001,
    )
    labels = (torch.arange(num_anchors, dtype=torch.long) % 1000).contiguous()
    anchors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_anchors, batch_size):
            end = min(start + batch_size, num_anchors)
            noise = noise_bank[start:end].to(device, non_blocking=True)
            y = labels[start:end].to(device, non_blocking=True)
            result = sample_grid(model, schedule, noise, y, solver="heun", nfe=reference_nfe)
            t0 = torch.full((result.samples.shape[0],), schedule.eps, device=device)
            eps0 = model(result.samples, t0, y)
            x0 = schedule.eps_to_x0(result.samples, t0, eps0)
            anchors.append(x0.detach().float().cpu())
            print(
                f"reference_anchor_progress model={model_name} nfe={reference_nfe} anchors={end}/{num_anchors}",
                flush=True,
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(anchors, dim=0)


def _build_anchor_latents(
    *,
    model: str,
    checkpoint: str,
    anchor_source: str,
    anchor_reference_nfe: int,
    anchor_noise_bank: str | None,
    num_anchors: int,
    batch_size: int,
    seed: int,
    device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if anchor_source == "gaussian":
        return torch.randn((num_anchors, 4, 32, 32), generator=generator)
    if anchor_source == "reference_x0":
        return _reference_x0_anchors(
            model_name=model,
            checkpoint=checkpoint,
            num_anchors=num_anchors,
            batch_size=batch_size,
            reference_nfe=anchor_reference_nfe,
            seed=seed,
            anchor_noise_bank=anchor_noise_bank,
            device=device,
        )
    raise ValueError(f"Unsupported anchor_source: {anchor_source}")


def estimate_decoder_spectral_gain(
    *,
    model: str,
    vae_dir: str,
    checkpoint: str,
    output_dir: Path,
    num_anchors: int,
    batch_size: int,
    latent_bands: int,
    image_bands: int,
    probes_per_band: int,
    epsilon: float,
    seed: int,
    anchor_source: str,
    anchor_reference_nfe: int,
    anchor_noise_bank: str | None,
    amp: bool,
) -> None:
    _load_torch_deps()
    set_seed(seed)
    device = default_device()
    vae = load_autoencoder_kl(vae_dir, device)
    latent_spec = radial_band_spec(32, 32, latent_bands, device)
    image_spec = radial_band_spec(256, 256, image_bands, device)
    gain = torch.zeros(latent_bands, image_bands, device=device)
    counts = torch.zeros(latent_bands, device=device)
    anchors = _build_anchor_latents(
        model=model,
        checkpoint=checkpoint,
        anchor_source=anchor_source,
        anchor_reference_nfe=anchor_reference_nfe,
        anchor_noise_bank=anchor_noise_bank,
        num_anchors=num_anchors,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    amp_enabled = bool(amp and device.type == "cuda")

    print(
        f"decoder_gain_start model={model} device={device} anchors={num_anchors} "
        f"batch_size={batch_size} probes_per_band={probes_per_band} epsilon={epsilon} "
        f"anchor_source={anchor_source} reference_nfe={anchor_reference_nfe} amp={amp_enabled}",
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, num_anchors, batch_size):
            end = min(start + batch_size, num_anchors)
            z = anchors[start:end].to(device, non_blocking=True)
            batch = z.shape[0]
            for latent_band in range(latent_bands):
                band_energy = torch.zeros(image_bands, device=device)
                for _probe in range(probes_per_band):
                    noise = torch.randn_like(z)
                    direction = _rms_normalize(_radial_project(noise, latent_spec, latent_band))
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        image_plus = _decode_unclamped(vae, z + epsilon * direction)
                        image_minus = _decode_unclamped(vae, z - epsilon * direction)
                    image_derivative = (image_plus.float() - image_minus.float()) / (2.0 * epsilon)
                    band_energy += fft_radial_band_energy(image_derivative, image_spec).sum(dim=0)
                gain[latent_band] += band_energy / float(probes_per_band)
                counts[latent_band] += batch
            print(f"decoder_gain_progress model={model} anchors={end}/{num_anchors}", flush=True)

    gain = gain / counts.clamp_min(1.0)[:, None]
    gain_np = gain.detach().cpu().numpy()
    row_sum = gain_np.sum(axis=1, keepdims=True)
    col_sum = gain_np.sum(axis=0, keepdims=True)
    total = float(gain_np.sum())
    rows = []
    for latent_band, latent_name in enumerate(_band_names(latent_bands)):
        for image_band, image_name in enumerate(_band_names(image_bands)):
            value = float(gain_np[latent_band, image_band])
            rows.append(
                {
                    "model": model,
                    "vae_dir": vae_dir,
                    "latent_band": latent_band,
                    "latent_band_name": latent_name,
                    "image_band": image_band,
                    "image_band_name": image_name,
                    "gain": value,
                    "row_transfer": value / max(float(row_sum[latent_band, 0]), 1e-12),
                    "col_transfer": value / max(float(col_sum[0, image_band]), 1e-12),
                    "fraction_total": value / max(total, 1e-12),
                    "num_anchors": num_anchors,
                    "probes_per_band": probes_per_band,
                    "epsilon": epsilon,
                    "anchor_source": anchor_source,
                    "anchor_reference_nfe": anchor_reference_nfe if anchor_source == "reference_x0" else "",
                    "anchor_checkpoint": checkpoint if anchor_source == "reference_x0" else "",
                    "estimator": "central_finite_difference_unit_rms_hutchinson",
                }
            )
    model_dir = ensure_dir(output_dir / model)
    _write_csv(model_dir / "decoder_spectral_gain_matrix.csv", rows)
    np.savez_compressed(
        model_dir / "decoder_spectral_gain_matrix.npz",
        model=np.asarray(model),
        vae_dir=np.asarray(vae_dir),
        gain=gain_np,
        latent_bands=np.arange(latent_bands),
        image_bands=np.arange(image_bands),
        anchor_source=np.asarray(anchor_source),
        anchor_reference_nfe=np.asarray(anchor_reference_nfe if anchor_source == "reference_x0" else -1),
        anchor_checkpoint=np.asarray(checkpoint if anchor_source == "reference_x0" else ""),
        estimator=np.asarray("central_finite_difference_unit_rms_hutchinson"),
    )
    _write_csv(
        model_dir / "decoder_spectral_gain_summary.csv",
        [
            _gain_summary_row(
                model,
                vae_dir,
                gain_np,
                args=None,
                anchor_source=anchor_source,
                anchor_reference_nfe=anchor_reference_nfe,
            )
        ],
    )
    _plot_gain_heatmap(model_dir, model, gain_np)
    print(f"decoder_gain_done model={model} output={model_dir}", flush=True)


def _gain_summary_row(
    model: str,
    vae_dir: str,
    gain: np.ndarray,
    args,
    anchor_source: str = "unknown",
    anchor_reference_nfe: int | str = "",
) -> dict[str, object]:
    latent_bands, image_bands = gain.shape
    high_latent = _high_band_start(latent_bands)
    high_image = _high_band_start(image_bands)
    total = float(gain.sum())
    diagonal = 0.0
    entangled = 0.0
    threshold = 1 if args is None else int(args.entangle_threshold)
    for latent_band in range(latent_bands):
        mapped = round(latent_band * (image_bands - 1) / max(latent_bands - 1, 1))
        for image_band in range(image_bands):
            value = float(gain[latent_band, image_band])
            if abs(mapped - image_band) <= threshold:
                diagonal += value
            else:
                entangled += value
    low_to_high = float(gain[:high_latent, high_image:].sum())
    high_to_low = float(gain[high_latent:, :high_image].sum())
    return {
        "model": model,
        "vae_dir": vae_dir,
        "gain_total": total,
        "diagonal_ratio": diagonal / max(total, 1e-12),
        "entangle_ratio": entangled / max(total, 1e-12),
        "low_latent_to_high_image_ratio": low_to_high / max(float(gain[:high_latent].sum()), 1e-12),
        "high_latent_to_low_image_ratio": high_to_low / max(float(gain[high_latent:].sum()), 1e-12),
        "high_image_gain_ratio": float(gain[:, high_image:].sum()) / max(total, 1e-12),
        "high_latent_gain_ratio": float(gain[high_latent:, :].sum()) / max(total, 1e-12),
        "latent_bands": latent_bands,
        "image_bands": image_bands,
        "entangle_threshold": threshold,
        "anchor_source": anchor_source,
        "anchor_reference_nfe": anchor_reference_nfe,
    }


def _plot_gain_heatmap(model_dir: Path, model: str, gain: np.ndarray) -> None:
    figure_dir = ensure_dir(model_dir / "figures")
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    image = ax.imshow(np.log10(gain + 1e-12), origin="upper", aspect="auto", cmap="magma")
    ax.set_title(f"{MODEL_LABELS.get(model, model)} decoder spectral gain")
    ax.set_xlabel("image radial frequency band")
    ax.set_ylabel("latent radial frequency band")
    fig.colorbar(image, ax=ax, label="log10 gain")
    fig.savefig(figure_dir / "decoder_spectral_gain_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "decoder_spectral_gain_heatmap.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _finite_corr_stats(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(xs) & np.isfinite(ys)
    x = xs[mask]
    y = ys[mask]
    if len(x) < 3 or x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    order_x = np.argsort(x)
    order_y = np.argsort(y)
    rank_x = np.empty_like(order_x, dtype=np.float64)
    rank_y = np.empty_like(order_y, dtype=np.float64)
    rank_x[order_x] = np.arange(len(x), dtype=np.float64)
    rank_y[order_y] = np.arange(len(y), dtype=np.float64)
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return pearson, spearman


def _load_gain(output_dir: Path, model: str) -> np.ndarray:
    path = output_dir / model / "decoder_spectral_gain_matrix.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing decoder gain matrix for {model}: {path}")
    return np.load(path, allow_pickle=False)["gain"].astype(np.float64)


def _score_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    return {
        (row.get("model", row.get("architecture", "")), row["solver"], int(row["nfe"])): row
        for row in rows
    }


def _resolve_latent_p2_model_dir(latent_p2_root: Path, model: str) -> Path:
    candidates = [
        latent_p2_root / f"imagenet256_p2_{model}",
        Path("outputs") / f"imagenet256_p2_{model}",
    ]
    for candidate in candidates:
        if (candidate / "trajectory_maps.npz").exists():
            return candidate
    return candidates[0]


def _load_latent_score_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    path = Path(args.latent_p2_scores)
    if path.exists():
        return _read_csv(path)
    rows: list[dict[str, str]] = []
    for model in args.models:
        model_dir = _resolve_latent_p2_model_dir(Path(args.latent_p2_root), model)
        score_path = model_dir / "trajectory_scores.csv"
        if score_path.exists():
            rows.extend(_read_csv(score_path))
    if not rows:
        raise FileNotFoundError(f"Could not find latent P2 scores at {path} or per-model trajectory_scores.csv files")
    return rows


def _load_decoded_score_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    path = Path(args.decoded_p2_scores)
    if path.exists():
        return _read_csv(path)
    rows: list[dict[str, str]] = []
    search_roots = [path.parent.parent if path.parent.name == "combined_analysis" else path.parent, Path("outputs")]
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for model in args.models:
            for score_path in sorted(root.glob(f"imagenet256_decoded_x0_p2_{model}_*/decoded_x0_frequency_scores.csv")):
                if score_path in seen:
                    continue
                seen.add(score_path)
                rows.extend(_read_csv(score_path))
    if not rows:
        raise FileNotFoundError(
            f"Could not find decoded P2 scores at {path} or per-solver decoded_x0_frequency_scores.csv files"
        )
    return rows


def _bin_count_weights(path: Path, solver: str, nfe: int, time_bins: int) -> np.ndarray:
    weights = np.zeros(time_bins, dtype=np.float64)
    if not path.exists():
        return np.ones(time_bins, dtype=np.float64) / float(time_bins)
    for row in _read_csv(path):
        if row["solver"] == solver and int(row["nfe"]) == nfe:
            weights[int(row["time_bin"])] = float(row["count"])
    if weights.sum() <= 0:
        active = np.ones(time_bins, dtype=np.float64)
        return active / active.sum()
    return weights / weights.sum()


def _weighted_latent_band_profile(latent_p2_root: Path, model: str, solver: str, nfe: int) -> np.ndarray:
    model_dir = _resolve_latent_p2_model_dir(latent_p2_root, model)
    maps = np.load(model_dir / "trajectory_maps.npz", allow_pickle=False)
    solvers = [str(item) for item in maps["solvers"]]
    nfes = [int(item) for item in maps["nfes"]]
    solver_index = solvers.index(solver)
    nfe_index = nfes.index(nfe)
    values = maps["x0_freq_error"][solver_index, nfe_index].astype(np.float64)
    weights = _bin_count_weights(model_dir / "trajectory_bins.csv", solver, nfe, values.shape[0])
    return (weights[:, None] * values).sum(axis=0)


def _decoder_weighted_predictors(
    latent_band_error: np.ndarray,
    gain: np.ndarray,
    entangle_threshold: int,
) -> dict[str, float]:
    latent_bands, image_bands = gain.shape
    high_latent = _high_band_start(latent_bands)
    high_image = _high_band_start(image_bands)
    weighted = latent_band_error[:, None] * gain
    total = float(weighted.sum())
    high = float(weighted[:, high_image:].sum())
    low_to_high = float(weighted[:high_latent, high_image:].sum())
    high_to_low = float(weighted[high_latent:, :high_image].sum())
    entangled = 0.0
    diagonal = 0.0
    for latent_band in range(latent_bands):
        mapped = round(latent_band * (image_bands - 1) / max(latent_bands - 1, 1))
        for image_band in range(image_bands):
            value = float(weighted[latent_band, image_band])
            if abs(mapped - image_band) <= entangle_threshold:
                diagonal += value
            else:
                entangled += value
    latent_total = float(latent_band_error.sum())
    latent_high = float(latent_band_error[high_latent:].sum())
    return {
        "latent_band_x0_freq_error": latent_total,
        "latent_band_x0_high_freq_error": latent_high,
        "latent_band_x0_high_freq_share": latent_high / max(latent_total, 1e-12),
        "decoder_weighted_p2_total": total,
        "decoder_weighted_p2_high_image": high,
        "decoder_weighted_p2_high_image_share": high / max(total, 1e-12),
        "decoder_weighted_p2_low_to_high": low_to_high,
        "decoder_weighted_p2_high_to_low": high_to_low,
        "decoder_weighted_p2_entangled": entangled,
        "decoder_weighted_p2_entangle_share": entangled / max(total, 1e-12),
        "decoder_weighted_p2_diagonal": diagonal,
    }


def analyze_latent_transport(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    latent_rows = _load_latent_score_rows(args)
    decoded_index = _score_index(_load_decoded_score_rows(args))
    score_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model in args.models:
        gain = _load_gain(output_dir, model)
        summary_rows.append(
            _gain_summary_row(
                model,
                _vae_dir_for_model(args, model),
                gain,
                args,
                anchor_source=args.anchor_source,
                anchor_reference_nfe=args.anchor_reference_nfe if args.anchor_source == "reference_x0" else "",
            )
        )
        for row in latent_rows:
            row_model = row.get("model", row.get("architecture", ""))
            if row_model != model:
                continue
            solver = row["solver"]
            nfe = int(row["nfe"])
            key = (model, solver, nfe)
            decoded = decoded_index.get(key, {})
            try:
                latent_band_error = _weighted_latent_band_profile(Path(args.latent_p2_root), model, solver, nfe)
            except (FileNotFoundError, ValueError) as exc:
                print(f"skip_missing_latent_map model={model} solver={solver} nfe={nfe} error={exc}", flush=True)
                continue
            predictors = _decoder_weighted_predictors(latent_band_error, gain, args.entangle_threshold)
            score_rows.append(
                {
                    "model": model,
                    "architecture": model,
                    "solver": solver,
                    "nfe": nfe,
                    "fid": float(row["fid"]),
                    "delta_fid": float(row["delta_fid"]),
                    "heun256_reference_fid": float(row["heun256_reference_fid"]),
                    "raw_latent_x0_freq_error": float(row["x0_freq_error"]),
                    "raw_latent_x0_high_freq_error": float(row["x0_high_freq_error"]),
                    "raw_latent_x0_high_freq_share": float(row["x0_high_freq_share"]),
                    "raw_latent_trajectory_drift": float(row["trajectory_drift"]),
                    "raw_latent_endpoint_drift": float(row["endpoint_drift"]),
                    "decoded_x0_freq_error": _float_or_nan(decoded.get("decoded_x0_freq_error")),
                    "decoded_x0_high_freq_error": _float_or_nan(decoded.get("decoded_x0_high_freq_error")),
                    "decoded_x0_high_freq_share": _float_or_nan(decoded.get("decoded_x0_high_freq_share")),
                    "decoded_x0_rms": _float_or_nan(decoded.get("decoded_x0_rms")),
                    **predictors,
                }
            )
    _write_csv(output_dir / "latent_transport_diffusability_scores.csv", score_rows)
    _write_csv(output_dir / "decoder_spectral_gain_summary.csv", summary_rows)
    correlation_rows = _correlation_rows(score_rows)
    _write_csv(output_dir / "latent_transport_diffusability_correlations.csv", correlation_rows)
    _plot_correlation_bars(output_dir, correlation_rows)
    _plot_predictor_scatters(output_dir, score_rows)
    _write_readme(output_dir, score_rows, correlation_rows, summary_rows)
    print(f"latent_transport_analysis_done output={output_dir}", flush=True)


def _float_or_nan(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _correlation_rows(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    predictors = [
        "raw_latent_x0_freq_error",
        "raw_latent_x0_high_freq_error",
        "raw_latent_trajectory_drift",
        "decoded_x0_freq_error",
        "decoded_x0_high_freq_error",
        "decoded_x0_rms",
        "decoder_weighted_p2_total",
        "decoder_weighted_p2_high_image",
        "decoder_weighted_p2_low_to_high",
        "decoder_weighted_p2_high_to_low",
        "decoder_weighted_p2_entangled",
        "decoder_weighted_p2_entangle_share",
    ]
    groups: dict[str, list[dict[str, object]]] = {"all": score_rows}
    for model in sorted({str(row["model"]) for row in score_rows}):
        groups[f"model:{model}"] = [row for row in score_rows if row["model"] == model]
    for solver in sorted({str(row["solver"]) for row in score_rows}):
        groups[f"solver:{solver}"] = [row for row in score_rows if row["solver"] == solver]
    rows: list[dict[str, object]] = []
    for group, group_rows in groups.items():
        y = np.asarray([float(row["delta_fid"]) for row in group_rows], dtype=np.float64)
        for predictor in predictors:
            x = np.asarray([float(row[predictor]) for row in group_rows], dtype=np.float64)
            pearson, spearman = _finite_corr_stats(x, y)
            rows.append(
                {
                    "group": group,
                    "predictor": predictor,
                    "target": "delta_fid_to_heun256",
                    "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if math.isfinite(pearson) else float("nan"),
                    "spearman_r": spearman,
                }
            )
    return rows


def _plot_correlation_bars(output_dir: Path, correlation_rows: list[dict[str, object]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    selected = [row for row in correlation_rows if row["group"] == "all" and int(row["n"]) >= 3]
    selected = sorted(selected, key=lambda row: float(row["pearson_r2"]) if math.isfinite(float(row["pearson_r2"])) else -1.0)
    labels = [str(row["predictor"]) for row in selected]
    values = [float(row["pearson_r2"]) for row in selected]
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.34 * len(labels))), constrained_layout=True)
    ax.barh(labels, values, color="#4C78A8")
    ax.set_xlabel("Pearson R^2 vs Delta FID to Heun256")
    ax.set_title("Latent-Transport Diffusability predictors")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(figure_dir / "ltd_predictor_correlation_bars.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "ltd_predictor_correlation_bars.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_predictor_scatters(output_dir: Path, score_rows: list[dict[str, object]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    predictors = [
        ("raw_latent_x0_freq_error", "Raw latent P2"),
        ("decoded_x0_freq_error", "Decoded-image P2"),
        ("decoder_weighted_p2_total", "Decoder-weighted P2"),
        ("decoder_weighted_p2_entangled", "Transport entangled P2"),
    ]
    fig, axes = plt.subplots(1, len(predictors), figsize=(5.0 * len(predictors), 4.0), constrained_layout=True)
    for ax, (predictor, title) in zip(axes, predictors):
        for solver in sorted({str(row["solver"]) for row in score_rows}):
            selected = [row for row in score_rows if row["solver"] == solver]
            ax.scatter(
                [float(row[predictor]) for row in selected],
                [float(row["delta_fid"]) for row in selected],
                s=42,
                alpha=0.85,
                color=SOLVER_COLORS.get(solver, "#777777"),
                label=SOLVER_LABELS.get(solver, solver),
            )
        x = np.asarray([float(row[predictor]) for row in score_rows], dtype=np.float64)
        y = np.asarray([float(row["delta_fid"]) for row in score_rows], dtype=np.float64)
        pearson, spearman = _finite_corr_stats(x, y)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) >= 3 and x[mask].std() > 1e-12:
            coef = np.polyfit(x[mask], y[mask], deg=1)
            xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#111827", linewidth=1.5, alpha=0.8)
        ax.set_title(f"{title}\nR2={pearson * pearson:.3f}, Spearman={spearman:.3f}")
        ax.set_xlabel(predictor)
        ax.set_ylabel("Delta FID to Heun256" if ax is axes[0] else "")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(figure_dir / "ltd_predictors_vs_quality.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "ltd_predictors_vs_quality.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    score_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    all_rows = [row for row in correlation_rows if row["group"] == "all" and int(row["n"]) >= 3]
    all_rows = sorted(
        all_rows,
        key=lambda row: float(row["pearson_r2"]) if math.isfinite(float(row["pearson_r2"])) else -1.0,
        reverse=True,
    )
    lines = [
        "# ImageNet256 Latent-Transport Diffusability Stage-1",
        "",
        "This diagnostic estimates decoder spectral gain and tests whether decoder-weighted latent P2 better predicts low-NFE quality.",
        "",
        "## Decoder Spectral Gain Summary",
        "",
        "| model | entangle_ratio | low_latent_to_high_image | high_latent_to_low_image | high_image_gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {entangle_ratio:.4f} | {low_latent_to_high_image_ratio:.4f} | "
            "{high_latent_to_low_image_ratio:.4f} | {high_image_gain_ratio:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Top Predictors",
            "",
            "| predictor | n | Pearson r | Pearson R2 | Spearman r |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in all_rows[:12]:
        lines.append(
            "| {predictor} | {n} | {pearson_r:.4f} | {pearson_r2:.4f} | {spearman_r:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `decoder_spectral_gain_summary.csv`: static decoder gain and entanglement summary.",
            "- `<model>/decoder_spectral_gain_matrix.csv`: full latent-band to image-band gain matrix.",
            "- `latent_transport_diffusability_scores.csv`: joined per model/solver/NFE predictors.",
            "- `latent_transport_diffusability_correlations.csv`: correlation table against Delta FID to Heun256.",
            "- `figures/`: heatmaps and predictor scatter/bar plots.",
            "",
            f"Rows analyzed: {len(score_rows)}.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if not args.skip_gain:
        for index, model in enumerate(args.models):
            estimate_decoder_spectral_gain(
                model=model,
                vae_dir=_vae_dir_for_model(args, model),
                checkpoint=_checkpoint_for_model(args, model),
                output_dir=output_dir,
                num_anchors=args.num_anchors,
                batch_size=args.batch_size,
                latent_bands=args.latent_bands,
                image_bands=args.image_bands,
                probes_per_band=args.probes_per_band,
                epsilon=args.epsilon,
                seed=args.seed + 1009 * index,
                anchor_source=args.anchor_source,
                anchor_reference_nfe=args.anchor_reference_nfe,
                anchor_noise_bank=args.anchor_noise_bank,
                amp=args.amp,
            )
    if not args.skip_analysis:
        analyze_latent_transport(args)


if __name__ == "__main__":
    main()
