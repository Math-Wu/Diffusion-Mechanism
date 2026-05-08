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
import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import fft_radial_band_energy, logsnr_bin_edges, radial_band_spec
from dm.eval_utils import load_or_create_noise_bank, noise_bank_id
from dm.imagenet256 import decode_latents, imagenet256_schedule, load_autoencoder_kl, load_imagenet256_model
from dm.utils import default_device, ensure_dir, set_seed

from eval_imagenet256_trajectory_mechanisms import (
    MODEL_LABELS,
    SOLVER_COLORS,
    SOLVER_LABELS,
    _batch_time,
    _finite_corr_stats,
    _heun256_reference_fid,
    _model_eps,
    _rms,
    _time_bin_index,
    high_band_slice,
    read_metrics,
    reference_heun_integrate,
    solver_step,
)
from run_imagenet256_pretrained_sweep import MODEL_SHAPES, SOLVERS, _default_noise_bank_path, _macro_intervals


LATENT_MODELS = ("dit_xl_2", "uvit_l_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ImageNet256 decoded-image P2 diagnostic for latent models. The solver and reference "
            "trajectories are computed in latent space, but x0 residual frequency energy is measured "
            "after VAE decoding to image space."
        )
    )
    parser.add_argument("--model", choices=LATENT_MODELS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--heun256_metrics")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=SOLVERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--num_paths", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--latent_scale_factor", type=float, default=0.18215)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_column", default="auto", choices=["auto", "delta_fid_to_heun256", "delta_fid", "fid"])
    parser.add_argument("--plot_only", action="store_true")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [dict(row) for row in _read_csv_rows(path)]


def _score_key(row: dict[str, object]) -> tuple[str, int]:
    return (str(row["solver"]), int(row["nfe"]))


@torch.no_grad()
def _decode_images(
    vae,
    latents: torch.Tensor,
    latent_scale_factor: float,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled and device.type == "cuda"):
        images = decode_latents(vae, latents, scale_factor=latent_scale_factor)
    return images.float()


def _zero_state(time_bins: int, freq_bands: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "counts": torch.zeros(time_bins, device=device),
        "decoded_x0_freq_error": torch.zeros(time_bins, freq_bands, device=device),
        "decoded_x0_rms": torch.zeros(time_bins, device=device),
    }


def _merge_state(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    for key, value in source.items():
        target[key] += value


@torch.no_grad()
def collect_decoded_x0_features(
    model,
    vae,
    schedule,
    initial_noise: torch.Tensor,
    labels: torch.Tensor,
    solver: str,
    nfe: int,
    time_bins: int,
    band_spec,
    ref_substeps: int,
    latent_scale_factor: float,
    amp_enabled: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = _zero_state(time_bins, band_spec.masks.shape[0], initial_noise.device)
    edges = logsnr_bin_edges(schedule, time_bins, initial_noise.device)
    intervals = _macro_intervals(solver, nfe)
    times = schedule.time_grid(intervals, initial_noise.device)
    x_solver = initial_noise
    x_reference = initial_noise.clone()
    remaining_nfe = nfe
    history: list[torch.Tensor] = []

    for index in range(intervals):
        t = _batch_time(times[index], initial_noise.shape[0])
        t_next = _batch_time(times[index + 1], initial_noise.shape[0])
        x_solver, remaining_nfe, history = solver_step(
            model,
            schedule,
            x_solver,
            t,
            t_next,
            labels,
            solver,
            remaining_nfe,
            history,
            amp_enabled,
            device,
        )
        x_reference = reference_heun_integrate(
            model, schedule, x_reference, t, t_next, labels, ref_substeps, amp_enabled, device
        )

        eps_solver = _model_eps(model, x_solver, t_next, labels, amp_enabled, device)
        eps_reference = _model_eps(model, x_reference, t_next, labels, amp_enabled, device)
        x0_solver = schedule.eps_to_x0(x_solver, t_next, eps_solver)
        x0_reference = schedule.eps_to_x0(x_reference, t_next, eps_reference)
        image_solver = _decode_images(vae, x0_solver, latent_scale_factor, amp_enabled, device)
        image_reference = _decode_images(vae, x0_reference, latent_scale_factor, amp_enabled, device)
        image_delta = image_solver - image_reference

        bin_index = _time_bin_index(edges, schedule.log_snr(t_next[0]))
        batch = initial_noise.shape[0]
        state["counts"][bin_index] += batch
        state["decoded_x0_rms"][bin_index] += _rms(image_delta).sum()
        state["decoded_x0_freq_error"][bin_index] += fft_radial_band_energy(image_delta, band_spec).sum(dim=0)

    return state


def _finalize_state(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    counts = state["counts"].clamp_min(1.0)
    return {
        "counts": state["counts"].detach().cpu().numpy(),
        "decoded_x0_rms": (state["decoded_x0_rms"] / counts).detach().cpu().numpy(),
        "decoded_x0_freq_error": (state["decoded_x0_freq_error"] / counts[:, None]).detach().cpu().numpy(),
    }


def _summarize_features(
    finalized: dict[str, np.ndarray],
    model_name: str,
    solver: str,
    nfe: int,
    metrics: dict[str, float],
    checkpoint: str,
    vae_dir: str,
    noise_id: str,
    num_paths: int,
    ref_substeps: int,
) -> dict[str, object]:
    counts = np.asarray(finalized["counts"], dtype=np.float64)
    weights = counts / max(float(counts.sum()), 1.0)
    decoded_freq = np.asarray(finalized["decoded_x0_freq_error"], dtype=np.float64)
    decoded_rms = np.asarray(finalized["decoded_x0_rms"], dtype=np.float64)
    high_slice = high_band_slice(decoded_freq.shape[1])
    total = float((weights[:, None] * decoded_freq).sum())
    high = float((weights[:, None] * decoded_freq[:, high_slice]).sum())
    return {
        "architecture": model_name,
        "model": model_name,
        "solver": solver,
        "nfe": nfe,
        "state_space": "decoded_image_from_latent_x0",
        "decoded_x0_freq_error": total,
        "decoded_x0_high_freq_error": high,
        "decoded_x0_high_freq_share": high / max(total, 1e-12),
        "decoded_x0_rms": float((weights * decoded_rms).sum()),
        "fid": metrics.get("fid", float("nan")),
        "delta_fid": metrics.get("delta_fid", float("nan")),
        "heun256_reference_fid": metrics.get("heun256_reference_fid", float("nan")),
        "wall_clock_sec": metrics.get("wall_clock_sec", float("nan")),
        "num_result_a_samples": int(metrics.get("num_samples", 0.0)),
        "num_paths": num_paths,
        "ref_substeps": ref_substeps,
        "checkpoint": checkpoint,
        "vae_dir": vae_dir,
        "noise_bank_id": noise_id,
    }


def _bin_rows(
    finalized: dict[str, np.ndarray],
    model_name: str,
    solver: str,
    nfe: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = np.asarray(finalized["counts"], dtype=np.float64)
    decoded_freq = np.asarray(finalized["decoded_x0_freq_error"], dtype=np.float64)
    decoded_rms = np.asarray(finalized["decoded_x0_rms"], dtype=np.float64)
    high_slice = high_band_slice(decoded_freq.shape[1])
    for time_bin in range(len(counts)):
        total = float(decoded_freq[time_bin].sum())
        rows.append(
            {
                "architecture": model_name,
                "model": model_name,
                "solver": solver,
                "nfe": nfe,
                "state_space": "decoded_image_from_latent_x0",
                "time_bin": time_bin,
                "count": int(counts[time_bin]),
                "decoded_x0_rms": float(decoded_rms[time_bin]),
                "decoded_x0_freq_error": total,
                "decoded_x0_high_freq_share": float(decoded_freq[time_bin, high_slice].sum() / max(total, 1e-12)),
            }
        )
    return rows


def _correlation_rows(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    predictors = ["decoded_x0_freq_error", "decoded_x0_high_freq_error", "decoded_x0_high_freq_share", "decoded_x0_rms"]
    groups: dict[str, list[dict[str, object]]] = {"all": score_rows}
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
                    "target": "delta_fid",
                    "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                    "spearman_r": spearman,
                }
            )
    return rows


def _plot_maps(output_dir: Path, model_name: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    maps = np.load(output_dir / "decoded_x0_frequency_maps.npz", allow_pickle=False)
    solvers = [str(item) for item in maps["solvers"]]
    nfes = [int(item) for item in maps["nfes"]]
    nfe_for_map = 8 if 8 in nfes else nfes[len(nfes) // 2]
    nfe_index = nfes.index(nfe_for_map)
    decoded_freq = maps["decoded_x0_freq_error"]
    fig, axes = plt.subplots(1, len(solvers), figsize=(3.5 * len(solvers), 3.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    last_image = None
    for si, solver in enumerate(solvers):
        data = decoded_freq[si, nfe_index]
        last_image = axes[si].imshow(np.log10(data + 1e-12), aspect="auto", origin="upper", cmap="magma")
        axes[si].set_title(f"{SOLVER_LABELS.get(solver, solver)}@{nfe_for_map}", fontsize=10)
        axes[si].set_xlabel("image radial frequency band")
        axes[si].set_ylabel("time bin" if si == 0 else "")
        axes[si].tick_params(labelsize=7)
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.78, label="log10 decoded-image x0 freq error")
    fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} decoded-image x0 frequency error")
    fig.savefig(figure_dir / "figure_decoded_x0_frequency_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure_decoded_x0_frequency_maps.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(output_dir: Path, score_rows: list[dict[str, object]], model_name: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    specs = [
        ("decoded_x0_high_freq_error", "Decoded-image high-frequency x0 error"),
        ("decoded_x0_freq_error", "Decoded-image total x0 frequency error"),
        ("decoded_x0_rms", "Decoded-image x0 RMS error"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.1 * len(specs), 4.1), constrained_layout=True)
    for ax, (key, title) in zip(axes, specs):
        for solver in SOLVERS:
            selected = [row for row in score_rows if row["solver"] == solver]
            if not selected:
                continue
            ax.scatter(
                [float(row[key]) for row in selected],
                [float(row["delta_fid"]) for row in selected],
                s=54,
                alpha=0.85,
                color=SOLVER_COLORS[solver],
                label=SOLVER_LABELS[solver],
            )
        x = np.asarray([float(row[key]) for row in score_rows], dtype=np.float64)
        y = np.asarray([float(row["delta_fid"]) for row in score_rows], dtype=np.float64)
        pearson, _ = _finite_corr_stats(x, y)
        if len(x) >= 2 and np.isfinite(x).all() and np.std(x) > 0:
            coef = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#111827", linewidth=1.8, alpha=0.8)
        ax.set_title(f"{title}\nPearson R2={pearson * pearson:.3f}", fontsize=11)
        ax.set_xlabel(key.replace("_", " "))
        ax.set_ylabel("Delta FID to Heun256" if ax is axes[0] else "")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} decoded-image P2 vs quality")
    fig.savefig(figure_dir / "figure_decoded_x0_predictors_vs_quality.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure_decoded_x0_predictors_vs_quality.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(output_dir: Path, score_rows: list[dict[str, object]], model_name: str) -> None:
    _plot_maps(output_dir, model_name)
    _plot_scatter(output_dir, score_rows, model_name)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.plot_only:
        score_rows = _load_existing(output_dir / "decoded_x0_frequency_scores.csv")
        plot_outputs(output_dir, score_rows, args.model)
        print(f"Regenerated decoded-image P2 figures under {output_dir / 'figures'}", flush=True)
        return

    set_seed(args.seed)
    device = default_device()
    amp_enabled = bool(args.amp and device.type == "cuda")
    schedule = imagenet256_schedule()
    model = load_imagenet256_model(args.model, args.checkpoint, device)
    vae = load_autoencoder_kl(args.vae_dir, device)
    shape = MODEL_SHAPES[args.model]
    noise_path = args.noise_bank or _default_noise_bank_path(args.model, args.seed + 777001, args.num_paths, shape)
    noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_paths, shape=shape, seed=args.seed + 777001)
    noise_id = noise_bank_id(noise_path)
    labels = (torch.arange(args.num_paths, dtype=torch.long) % 1000).contiguous()
    band_spec = radial_band_spec(256, 256, args.freq_bands, device)
    heun256_fid = _heun256_reference_fid(args.heun256_metrics)
    metrics_table = read_metrics(Path(args.metrics_csv), args.model, heun256_fid, args.target_column)
    print(
        f"imagenet256_decoded_x0_p2_start model={args.model} device={device} "
        f"num_paths={args.num_paths} batch_size={args.batch_size} amp={amp_enabled} vae={args.vae_dir}",
        flush=True,
    )

    score_path = output_dir / "decoded_x0_frequency_scores.csv"
    bin_path = output_dir / "decoded_x0_frequency_bins.csv"
    score_rows: list[dict[str, object]] = _load_existing(score_path)
    bin_rows: list[dict[str, object]] = _load_existing(bin_path)
    completed = {_score_key(row) for row in score_rows}
    decoded_maps: dict[tuple[str, int], np.ndarray] = {}
    rms_maps: dict[tuple[str, int], np.ndarray] = {}
    map_path = output_dir / "decoded_x0_frequency_maps.npz"
    if map_path.exists():
        payload = np.load(map_path, allow_pickle=False)
        old_solvers = [str(item) for item in payload["solvers"]]
        old_nfes = [int(item) for item in payload["nfes"]]
        for si, solver in enumerate(old_solvers):
            for ni, nfe in enumerate(old_nfes):
                decoded_maps[(solver, nfe)] = payload["decoded_x0_freq_error"][si, ni]
                rms_maps[(solver, nfe)] = payload["decoded_x0_rms"][si, ni]

    solvers = list(args.solvers)
    nfes = list(args.nfe)
    for solver in solvers:
        for nfe in nfes:
            key = (solver, nfe)
            if key in completed:
                print(f"skip_completed model={args.model} solver={solver} nfe={nfe}", flush=True)
                continue
            print(f"BEGIN decoded_x0_p2 model={args.model} solver={solver} nfe={nfe}", flush=True)
            merged = _zero_state(args.time_bins, args.freq_bands, device)
            for start in range(0, noise_bank.shape[0], args.batch_size):
                end = min(start + args.batch_size, noise_bank.shape[0])
                noise = noise_bank[start:end].to(device, non_blocking=True)
                y = labels[start:end].to(device, non_blocking=True)
                batch_state = collect_decoded_x0_features(
                    model,
                    vae,
                    schedule,
                    noise,
                    y,
                    solver,
                    nfe,
                    args.time_bins,
                    band_spec,
                    args.ref_substeps,
                    args.latent_scale_factor,
                    amp_enabled,
                    device,
                )
                _merge_state(merged, batch_state)
                print(f"progress model={args.model} solver={solver} nfe={nfe} paths={end}/{noise_bank.shape[0]}", flush=True)
            finalized = _finalize_state(merged)
            decoded_maps[key] = np.asarray(finalized["decoded_x0_freq_error"], dtype=np.float64)
            rms_maps[key] = np.asarray(finalized["decoded_x0_rms"], dtype=np.float64)
            row = _summarize_features(
                finalized,
                args.model,
                solver,
                nfe,
                metrics_table.get(key, {}),
                args.checkpoint,
                args.vae_dir,
                noise_id,
                args.num_paths,
                args.ref_substeps,
            )
            score_rows.append(row)
            bin_rows.extend(_bin_rows(finalized, args.model, solver, nfe))
            _write_csv(score_path, score_rows)
            _write_csv(bin_path, bin_rows)
            _write_csv(output_dir / "decoded_x0_frequency_correlations.csv", _correlation_rows(score_rows))

            decoded_array = np.asarray(
                [
                    [decoded_maps.get((s, n), np.full((args.time_bins, args.freq_bands), np.nan)) for n in nfes]
                    for s in solvers
                ]
            )
            rms_array = np.asarray(
                [[rms_maps.get((s, n), np.full(args.time_bins, np.nan)) for n in nfes] for s in solvers]
            )
            np.savez_compressed(
                map_path,
                architecture=np.asarray(args.model),
                model=np.asarray(args.model),
                state_space=np.asarray("decoded_image_from_latent_x0"),
                solvers=np.asarray(solvers),
                nfes=np.asarray(nfes),
                decoded_x0_freq_error=decoded_array,
                decoded_x0_rms=rms_array,
            )
            print(
                f"DONE decoded_x0_p2 model={args.model} solver={solver} nfe={nfe} "
                f"decoded_x0_high_freq_error={float(row['decoded_x0_high_freq_error']):.6e} "
                f"decoded_x0_rms={float(row['decoded_x0_rms']):.6e} "
                f"delta_fid={float(row['delta_fid']):.6f}",
                flush=True,
            )

    _write_csv(output_dir / "decoded_x0_frequency_correlations.csv", _correlation_rows(score_rows))
    plot_outputs(output_dir, score_rows, args.model)
    print(f"Wrote decoded-image x0 P2 diagnostic outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
