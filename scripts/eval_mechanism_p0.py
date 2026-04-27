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

from dm.analysis import fft_radial_band_energy, logsnr_bin_centers, radial_band_spec
from dm.data import build_cifar10_loaders
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.samplers import SAMPLERS
from dm.utils import default_device, ensure_dir, set_seed


ARCH_LABELS = {"unet": "U-Net", "uvit": "U-ViT", "dit": "DiT"}
SOLVER_LABELS = {"ddim": "DDIM", "heun": "Heun", "dpmpp": "DPM++", "unipc": "UniPC"}
SOLVER_COLORS = {"ddim": "#9a3412", "heun": "#2563eb", "dpmpp": "#15803d", "unipc": "#7e22ce"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P0 mechanism metrics for CIFAR Result A.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--metrics_csv", default="outputs/result_a_cifar_medium/metrics.csv")
    parser.add_argument("--output_dir", default="outputs/mechanism_p0_cifar_medium")
    parser.add_argument("--checkpoint_name", default="last.pt")
    parser.add_argument("--solvers", nargs="+", default=["ddim", "heun", "dpmpp", "unipc"], choices=SAMPLERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[8, 15, 20, 50])
    parser.add_argument("--num_data", type=int, default=512)
    parser.add_argument(
        "--num_noise",
        type=int,
        default=128,
        help="Legacy name: number of held-out q-path examples for fixed-bin local-error probes.",
    )
    parser.add_argument("--num_error", type=int, dest="num_noise")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank", help=argparse.SUPPRESS)
    parser.add_argument("--raw_weights", action="store_true")
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


@torch.no_grad()
def ddim_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    eps = model(x, t)
    x0 = schedule.eps_to_x0(x, t, eps)
    alpha_next, sigma_next = schedule.alpha_sigma(t_next)
    while alpha_next.ndim < x.ndim:
        alpha_next = alpha_next[..., None]
        sigma_next = sigma_next[..., None]
    return alpha_next * x0 + sigma_next * eps


@torch.no_grad()
def heun_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, use_corrector: bool = True) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
    if not use_corrector:
        return x + dt * drift
    x_euler = x + dt * drift
    eps_next = model(x_euler, t_next)
    drift_next = schedule.drift(x_euler, t_next, eps_next)
    return x + 0.5 * dt * (drift + drift_next)


@torch.no_grad()
def reference_heun_integrate(model, schedule, x: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor, substeps: int) -> torch.Tensor:
    lambda_start = schedule.log_snr(t_start[0])
    lambda_end = schedule.log_snr(t_end[0])
    lambdas = torch.linspace(lambda_start, lambda_end, substeps + 1, device=x.device)
    times = schedule.inverse_log_snr(lambdas)
    current = x
    for index in range(substeps):
        t = _batch_time(times[index], x.shape[0])
        t_next = _batch_time(times[index + 1], x.shape[0])
        current = heun_step(model, schedule, current, t, t_next, use_corrector=True)
    return current


def _logsnr_bounds(schedule, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t_hi = torch.tensor(1.0 - schedule.eps, device=device)
    t_lo = torch.tensor(schedule.eps, device=device)
    return schedule.log_snr(t_hi), schedule.log_snr(t_lo)


def _macro_intervals(solver: str, nfe: int) -> int:
    return max(1, (nfe + 1) // 2) if solver == "heun" else max(1, nfe)


@torch.no_grad()
def run_fixed_bin_solver_error_map(
    model,
    schedule,
    val_loader,
    solver: str,
    nfe: int,
    device: torch.device,
    num_examples: int,
    time_bins: int,
    band_spec,
    ref_substeps: int,
    seed: int,
) -> np.ndarray:
    if solver not in {"ddim", "heun", "dpmpp", "unipc"}:
        raise ValueError(f"Unsupported P0 solver: {solver}")
    lambda_hi, lambda_lo = _logsnr_bounds(schedule, device)
    edges = torch.linspace(lambda_hi, lambda_lo, time_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lambda_step = (lambda_lo - lambda_hi) / _macro_intervals(solver, nfe)
    sums = torch.zeros(time_bins, band_spec.masks.shape[0], device=device)
    counts = torch.zeros(time_bins, device=device)
    max_order = 2 if solver == "dpmpp" else 3
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    seen = 0

    for batch in val_loader:
        if seen >= num_examples:
            break
        x0 = batch[0].to(device)
        if seen + x0.shape[0] > num_examples:
            x0 = x0[: num_examples - seen]
        seen += x0.shape[0]
        path_noise = torch.randn(x0.shape, device=device, generator=generator)

        for bin_index, lambda_start in enumerate(centers):
            lambda_end = torch.minimum(lambda_start + lambda_step, lambda_lo)
            t_start_scalar = schedule.inverse_log_snr(lambda_start)
            t_end_scalar = schedule.inverse_log_snr(lambda_end)
            t_start = _batch_time(t_start_scalar, x0.shape[0])
            t_end = _batch_time(t_end_scalar, x0.shape[0])
            x_start = schedule.q_sample(x0, t_start, path_noise)
            x_ref = reference_heun_integrate(model, schedule, x_start, t_start, t_end, ref_substeps)

            if solver == "ddim":
                x_next = ddim_step(model, schedule, x_start, t_start, t_end)
            elif solver == "heun":
                x_next = heun_step(model, schedule, x_start, t_start, t_end, use_corrector=True)
            else:
                history: list[torch.Tensor] = []
                for history_index in range(max_order):
                    lambda_hist = lambda_start - history_index * lambda_step
                    if lambda_hist < lambda_hi:
                        break
                    t_hist_scalar = schedule.inverse_log_snr(lambda_hist)
                    t_hist = _batch_time(t_hist_scalar, x0.shape[0])
                    x_hist = schedule.q_sample(x0, t_hist, path_noise)
                    eps_hist = model(x_hist, t_hist)
                    history.append(schedule.drift(x_hist, t_hist, eps_hist))
                coeffs = {
                    1: (1.0,),
                    2: (1.5, -0.5),
                    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
                }[len(history)]
                update = torch.zeros_like(x_start)
                for coeff, old_drift in zip(coeffs, history):
                    update = update + coeff * old_drift
                dt = (t_end - t_start).view(x0.shape[0], *([1] * (x0.ndim - 1)))
                x_next = x_start + dt * update

            energy = fft_radial_band_energy(x_next - x_ref, band_spec)
            sums[bin_index] += energy.sum(dim=0)
            counts[bin_index] += x0.shape[0]

    if seen == 0:
        raise RuntimeError("No validation samples were available for solver error map computation")
    return (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()


@torch.no_grad()
def compute_difficulty_map(model, schedule, val_loader, device, num_data: int, time_bins: int, band_spec, seed: int) -> np.ndarray:
    times = logsnr_bin_centers(schedule, time_bins, device)
    sums = torch.zeros(time_bins, band_spec.masks.shape[0], device=device)
    counts = torch.zeros(time_bins, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    seen = 0
    for batch in val_loader:
        x0 = batch[0].to(device)
        if seen >= num_data:
            break
        if seen + x0.shape[0] > num_data:
            x0 = x0[: num_data - seen]
        seen += x0.shape[0]
        for index, scalar_t in enumerate(times):
            t = _batch_time(scalar_t, x0.shape[0])
            noise = torch.randn(x0.shape, device=device, generator=generator)
            x_t = schedule.q_sample(x0, t, noise)
            eps = model(x_t, t)
            x0_hat = schedule.eps_to_x0(x_t, t, eps)
            energy = fft_radial_band_energy(x0_hat - x0, band_spec)
            sums[index] += energy.sum(dim=0)
            counts[index] += x0.shape[0]
    if seen == 0:
        raise RuntimeError("No validation samples were available for difficulty map computation")
    return (sums / counts.clamp_min(1.0)[:, None]).detach().cpu().numpy()


def read_result_a_metrics(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    table = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["solver"] not in {"ddim", "heun", "dpmpp", "unipc"}:
                continue
            table[(row["architecture"], row["solver"], int(row["nfe"]))] = {
                "fid": float(row["fid"]),
                "delta_fid": float(row["delta_fid"]),
                "wall_clock_sec": float(row["wall_clock_sec"]),
                "num_samples": float(row.get("num_samples", 10000)),
            }
    return table


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr_stats(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    if len(xs) < 3 or np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    spearman = float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])
    return pearson, spearman


def write_score_tables(output_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    score_path = output_dir / "mechanism_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    corr_rows = []
    groups: dict[str, list[dict[str, float | int | str]]] = {"all": rows}
    for arch in sorted({str(row["architecture"]) for row in rows}):
        groups[arch] = [row for row in rows if row["architecture"] == arch]
    for group, group_rows in groups.items():
        y = np.asarray([float(row["delta_fid"]) for row in group_rows])
        for key in ("total_error", "difficulty_weighted_error", "overlap_score"):
            x = np.asarray([float(row[key]) for row in group_rows])
            pearson, spearman = corr_stats(x, y)
            corr_rows.append(
                {
                    "group": group,
                    "predictor": key,
                    "n": len(group_rows),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                    "spearman_r": spearman,
                }
            )
    with (output_dir / "mechanism_correlations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corr_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corr_rows)


def plot_heatmap(ax, data: np.ndarray, title: str) -> None:
    image = ax.imshow(np.log10(data + 1e-12), aspect="auto", origin="upper", cmap="magma")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Radial frequency band")
    ax.set_ylabel("time bin: noise -> data")
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.tick_params(labelsize=7)
    return image


def plot_outputs(output_dir: Path, architectures: list[str], solvers: list[str], nfes: list[int], difficulty, solver_error, rows) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    fig, axes = plt.subplots(1, len(architectures), figsize=(4.2 * len(architectures), 3.8))
    axes = np.atleast_1d(axes)
    last_image = None
    for ai, arch in enumerate(architectures):
        last_image = plot_heatmap(axes[ai], difficulty[ai], f"{ARCH_LABELS.get(arch, arch)} difficulty")
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.72, label="log10 FFT-band x0 error")
    fig.suptitle("Figure 5. Architecture Difficulty Maps", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure5_difficulty_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure5_difficulty_maps.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)

    nfe_for_error = 20 if 20 in nfes else nfes[len(nfes) // 2]
    fig, axes = plt.subplots(len(architectures), len(solvers), figsize=(3.5 * len(solvers), 3.2 * len(architectures)))
    axes = np.atleast_2d(axes)
    last_image = None
    nfe_index = nfes.index(nfe_for_error)
    for ai, arch in enumerate(architectures):
        for si, solver in enumerate(solvers):
            data = solver_error[ai, si, nfe_index]
            last_image = plot_heatmap(axes[ai, si], data, f"{ARCH_LABELS.get(arch, arch)} {SOLVER_LABELS.get(solver, solver)}@{nfe_for_error}")
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.72, label="log10 local solver error")
    fig.suptitle("Figure 6. Solver Error Maps", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure6_solver_error_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure6_solver_error_maps.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2))
    for ax, key, title in [
        (axes[0], "total_error", "Total solver error"),
        (axes[1], "difficulty_weighted_error", "Difficulty-weighted error"),
        (axes[2], "overlap_score", "Normalized overlap score"),
    ]:
        for solver in solvers:
            selected = [row for row in rows if row["solver"] == solver]
            x = np.asarray([float(row[key]) for row in selected])
            y = np.asarray([float(row["delta_fid"]) for row in selected])
            ax.scatter(x, y, s=46, color=SOLVER_COLORS[solver], alpha=0.82, label=SOLVER_LABELS[solver])
        x_all = np.asarray([float(row[key]) for row in rows])
        y_all = np.asarray([float(row["delta_fid"]) for row in rows])
        pearson, _ = corr_stats(x_all, y_all)
        if len(x_all) >= 2 and np.std(x_all) > 0:
            coef = np.polyfit(x_all, y_all, deg=1)
            xs = np.linspace(float(x_all.min()), float(x_all.max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#111827", linewidth=1.8, alpha=0.8)
        ax.set_title(f"{title} vs Delta FID\nPearson R2={pearson * pearson:.3f}", fontsize=12, weight="bold")
        ax.set_xlabel(key.replace("_", " "))
        ax.set_ylabel("Delta FID")
        ax.grid(True, alpha=0.25)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Figure 7. Does Overlap Explain Quality?", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure7_overlap_explains_quality.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure7_overlap_explains_quality.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.run_dirs):
        raise ValueError("--configs and --run_dirs must have the same length")
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    metrics_table = read_result_a_metrics(Path(args.metrics_csv))
    architectures: list[str] = []
    difficulty_maps = []
    solver_maps = []
    score_rows: list[dict[str, float | int | str]] = []

    for config_path, run_dir in zip(args.configs, args.run_dirs):
        ckpt = checkpoint_path_for_run(run_dir, name=args.checkpoint_name)
        model, config, checkpoint = load_model_from_checkpoint(config_path, ckpt, device, use_ema=not args.raw_weights)
        config["training"]["batch_size"] = args.batch_size
        architecture = config["model"]["architecture"]
        architectures.append(architecture)
        schedule = build_schedule(config)
        _, val_loader = build_cifar10_loaders(config, download=False)
        image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
        band_spec = radial_band_spec(image_size, image_size, args.freq_bands, device)
        print(f"BEGIN difficulty architecture={architecture}", flush=True)
        difficulty = compute_difficulty_map(
            model,
            schedule,
            val_loader,
            device,
            args.num_data,
            args.time_bins,
            band_spec,
            args.seed + 17,
        )
        difficulty_maps.append(difficulty)
        d_norm = difficulty / max(float(difficulty.sum()), 1e-12)

        arch_solver_maps = []
        checkpoint_id = f"step_{int(checkpoint.get('step', -1))}_images_{int(checkpoint.get('images_seen', -1))}"
        for solver in args.solvers:
            solver_nfe_maps = []
            for nfe in args.nfe:
                print(f"BEGIN solver_error architecture={architecture} solver={solver} nfe={nfe}", flush=True)
                error_map = run_fixed_bin_solver_error_map(
                    model,
                    schedule,
                    val_loader,
                    solver,
                    nfe,
                    device,
                    args.num_noise,
                    args.time_bins,
                    band_spec,
                    args.ref_substeps,
                    args.seed + 31,
                )
                solver_nfe_maps.append(error_map)
                e_norm = error_map / max(float(error_map.sum()), 1e-12)
                overlap = float((d_norm * e_norm).sum())
                total_error = float(error_map.sum())
                difficulty_weighted_error = float((d_norm * error_map).sum())
                metrics = metrics_table.get((architecture, solver, nfe), {})
                score_rows.append(
                    {
                        "architecture": architecture,
                        "solver": solver,
                        "nfe": nfe,
                        "overlap_score": overlap,
                        "normalized_overlap_score": overlap,
                        "total_error": total_error,
                        "difficulty_weighted_error": difficulty_weighted_error,
                        "fid": metrics.get("fid", float("nan")),
                        "delta_fid": metrics.get("delta_fid", float("nan")),
                        "wall_clock_sec": metrics.get("wall_clock_sec", float("nan")),
                        "num_result_a_samples": int(metrics.get("num_samples", 0.0)),
                        "num_data": args.num_data,
                        "num_noise": args.num_noise,
                        "time_bins": args.time_bins,
                        "freq_bands": args.freq_bands,
                        "ref_substeps": args.ref_substeps,
                        "local_error_probe": "fixed_logsnr_bin_q_path",
                        "macro_intervals": _macro_intervals(solver, nfe),
                        "checkpoint": str(ckpt),
                        "checkpoint_id": checkpoint_id,
                    }
                )
                print(
                    f"DONE architecture={architecture} solver={solver} nfe={nfe} "
                    f"overlap={overlap:.6e} total_error={total_error:.6e} "
                    f"difficulty_weighted_error={difficulty_weighted_error:.6e} "
                    f"delta_fid={metrics.get('delta_fid', float('nan')):.6f}",
                    flush=True,
                )
            arch_solver_maps.append(solver_nfe_maps)
        solver_maps.append(arch_solver_maps)

    difficulty_array = np.asarray(difficulty_maps, dtype=np.float64)
    solver_error_array = np.asarray(solver_maps, dtype=np.float64)
    np.savez_compressed(
        output_dir / "mechanism_maps.npz",
        architectures=np.asarray(architectures),
        solvers=np.asarray(args.solvers),
        nfes=np.asarray(args.nfe),
        difficulty=difficulty_array,
        solver_error=solver_error_array,
    )
    write_score_tables(output_dir, score_rows)
    plot_outputs(output_dir, architectures, args.solvers, args.nfe, difficulty_array, solver_error_array, score_rows)
    print(f"Wrote P0 mechanism outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
