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
from torch.nn import functional as F

from _bootstrap import add_src_to_path

add_src_to_path()

from dm.analysis import fft_radial_band_energy, logsnr_bin_edges, radial_band_spec
from dm.eval_utils import default_noise_bank_path, load_or_create_noise_bank, noise_bank_id, noise_batches
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.samplers.ode import AB_COEFFS
from dm.utils import default_device, ensure_dir, set_seed


ARCH_LABELS = {"unet": "U-Net", "uvit": "U-ViT", "dit": "DiT"}
ARCH_COLORS = {"unet": "#0f766e", "uvit": "#9333ea", "dit": "#ea580c"}
SOLVER_LABELS = {"ddim": "DDIM", "heun": "Heun", "dpmpp": "DPM++", "unipc": "UniPC"}
SOLVER_COLORS = {"ddim": "#9a3412", "heun": "#2563eb", "dpmpp": "#15803d", "unipc": "#7e22ce"}
SUPPORTED_SOLVERS = ("ddim", "heun", "dpmpp", "unipc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate P2 trajectory mechanisms: directional alignment, frequency-wise denoising error, "
            "and trajectory drift against a high-accuracy Heun reference path."
        )
    )
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--run_dirs", nargs="+")
    parser.add_argument("--metrics_csv", default="outputs/result_a_cifar_medium/metrics.csv")
    parser.add_argument("--output_dir", default="outputs/trajectory_mechanisms_cifar_medium")
    parser.add_argument("--checkpoint_name", default="last.pt")
    parser.add_argument("--solvers", nargs="+", default=list(SUPPORTED_SOLVERS), choices=SUPPORTED_SOLVERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[8, 15, 20, 50])
    parser.add_argument("--num_paths", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--raw_weights", action="store_true")
    parser.add_argument("--plot_only", action="store_true", help="Regenerate figures from existing trajectory outputs.")
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).square().mean(dim=1).sqrt()


def _macro_intervals(solver: str, nfe: int) -> int:
    return max(1, (nfe + 1) // 2) if solver == "heun" else max(1, nfe)


def _time_bin_index(edges: torch.Tensor, log_snr: torch.Tensor) -> int:
    value = log_snr.detach().clamp(edges[0], edges[-1])
    index = torch.searchsorted(edges, value, right=True).item() - 1
    return max(0, min(edges.numel() - 2, int(index)))


def _finite_corr_stats(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(xs) & np.isfinite(ys)
    x = xs[mask]
    y = ys[mask]
    if len(x) < 3 or x.std() == 0.0 or y.std() == 0.0:
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


@torch.no_grad()
def heun_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
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
        current = heun_step(model, schedule, current, t, t_next)
    return current


@torch.no_grad()
def ddim_solver_step(model, schedule, x: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    eps = model(x, t)
    x0 = schedule.eps_to_x0(x, t, eps)
    alpha_next, sigma_next = schedule.alpha_sigma(t_next)
    while alpha_next.ndim < x.ndim:
        alpha_next = alpha_next[..., None]
        sigma_next = sigma_next[..., None]
    return alpha_next * x0 + sigma_next * eps


@torch.no_grad()
def heun_solver_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    remaining_nfe: int,
) -> tuple[torch.Tensor, int]:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
    remaining_nfe -= 1
    if remaining_nfe > 0:
        x_euler = x + dt * drift
        eps_next = model(x_euler, t_next)
        drift_next = schedule.drift(x_euler, t_next, eps_next)
        x = x + 0.5 * dt * (drift + drift_next)
        remaining_nfe -= 1
    else:
        x = x + dt * drift
    return x, remaining_nfe


@torch.no_grad()
def multistep_solver_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    history: list[torch.Tensor],
    max_order: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = model(x, t)
    drift = schedule.drift(x, t, eps)
    history.insert(0, drift)
    order = min(max_order, len(history))
    update = torch.zeros_like(x)
    for coeff, old_drift in zip(AB_COEFFS[order], history):
        update = update + coeff * old_drift
    return x + dt * update, history[:max_order]


@torch.no_grad()
def solver_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    solver: str,
    remaining_nfe: int,
    history: list[torch.Tensor],
) -> tuple[torch.Tensor, int, list[torch.Tensor]]:
    if solver == "ddim":
        return ddim_solver_step(model, schedule, x, t, t_next), remaining_nfe - 1, history
    if solver == "heun":
        x_next, remaining_nfe = heun_solver_step(model, schedule, x, t, t_next, remaining_nfe)
        return x_next, remaining_nfe, history
    if solver == "dpmpp":
        x_next, history = multistep_solver_step(model, schedule, x, t, t_next, history, max_order=2)
        return x_next, remaining_nfe - 1, history
    if solver == "unipc":
        x_next, history = multistep_solver_step(model, schedule, x, t, t_next, history, max_order=3)
        return x_next, remaining_nfe - 1, history
    raise ValueError(f"Unsupported P2 solver: {solver}")


def zero_feature_state(time_bins: int, freq_bands: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "counts": torch.zeros(time_bins, device=device),
        "eps_alignment": torch.zeros(time_bins, device=device),
        "drift_alignment": torch.zeros(time_bins, device=device),
        "eps_delta_rms": torch.zeros(time_bins, device=device),
        "drift_delta_rms": torch.zeros(time_bins, device=device),
        "trajectory_drift": torch.zeros(time_bins, device=device),
        "endpoint_drift": torch.zeros((), device=device),
        "endpoint_count": torch.zeros((), device=device),
        "eps_freq_error": torch.zeros(time_bins, freq_bands, device=device),
        "x0_freq_error": torch.zeros(time_bins, freq_bands, device=device),
    }


def merge_feature_state(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    for key, value in source.items():
        target[key] += value


@torch.no_grad()
def collect_solver_trajectory_features(
    model,
    schedule,
    initial_noise: torch.Tensor,
    solver: str,
    nfe: int,
    time_bins: int,
    band_spec,
    ref_substeps: int,
) -> dict[str, torch.Tensor]:
    state = zero_feature_state(time_bins, band_spec.masks.shape[0], initial_noise.device)
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
            solver,
            remaining_nfe,
            history,
        )
        x_reference = reference_heun_integrate(model, schedule, x_reference, t, t_next, ref_substeps)

        eps_solver = model(x_solver, t_next)
        eps_reference = model(x_reference, t_next)
        drift_solver = schedule.drift(x_solver, t_next, eps_solver)
        drift_reference = schedule.drift(x_reference, t_next, eps_reference)
        x0_solver = schedule.eps_to_x0(x_solver, t_next, eps_solver)
        x0_reference = schedule.eps_to_x0(x_reference, t_next, eps_reference)

        bin_index = _time_bin_index(edges, schedule.log_snr(t_next[0]))
        batch = initial_noise.shape[0]
        state["counts"][bin_index] += batch
        state["eps_alignment"][bin_index] += F.cosine_similarity(eps_solver.flatten(1), eps_reference.flatten(1), dim=1).sum()
        state["drift_alignment"][bin_index] += F.cosine_similarity(drift_solver.flatten(1), drift_reference.flatten(1), dim=1).sum()
        state["eps_delta_rms"][bin_index] += _rms(eps_solver - eps_reference).sum()
        state["drift_delta_rms"][bin_index] += _rms(drift_solver - drift_reference).sum()
        state["trajectory_drift"][bin_index] += _rms(x_solver - x_reference).sum()
        state["eps_freq_error"][bin_index] += fft_radial_band_energy(eps_solver - eps_reference, band_spec).sum(dim=0)
        state["x0_freq_error"][bin_index] += fft_radial_band_energy(x0_solver - x0_reference, band_spec).sum(dim=0)

    state["endpoint_drift"] += _rms(x_solver - x_reference).sum()
    state["endpoint_count"] += initial_noise.shape[0]
    return state


def finalize_state(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray | float]:
    counts = state["counts"].clamp_min(1.0)
    endpoint_count = state["endpoint_count"].clamp_min(1.0)
    finalized: dict[str, np.ndarray | float] = {
        "counts": state["counts"].detach().cpu().numpy(),
        "eps_alignment": (state["eps_alignment"] / counts).detach().cpu().numpy(),
        "drift_alignment": (state["drift_alignment"] / counts).detach().cpu().numpy(),
        "eps_delta_rms": (state["eps_delta_rms"] / counts).detach().cpu().numpy(),
        "drift_delta_rms": (state["drift_delta_rms"] / counts).detach().cpu().numpy(),
        "trajectory_drift": (state["trajectory_drift"] / counts).detach().cpu().numpy(),
        "eps_freq_error": (state["eps_freq_error"] / counts[:, None]).detach().cpu().numpy(),
        "x0_freq_error": (state["x0_freq_error"] / counts[:, None]).detach().cpu().numpy(),
        "endpoint_drift": float((state["endpoint_drift"] / endpoint_count).detach().cpu()),
    }
    return finalized


def read_result_a_metrics(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    table: dict[tuple[str, str, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            solver = row["solver"]
            if solver not in SUPPORTED_SOLVERS:
                continue
            table[(row["architecture"], solver, int(row["nfe"]))] = {
                "fid": float(row["fid"]),
                "delta_fid": float(row["delta_fid"]),
                "wall_clock_sec": float(row["wall_clock_sec"]),
                "num_samples": float(row.get("num_samples", 10000)),
            }
    return table


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def high_band_slice(num_bands: int) -> slice:
    start = max(0, int(math.ceil(num_bands * 0.625)))
    return slice(start, num_bands)


def summarize_features(
    finalized: dict[str, np.ndarray | float],
    architecture: str,
    solver: str,
    nfe: int,
    metrics: dict[str, float],
    checkpoint: Path,
    checkpoint_id: str,
    noise_id: str,
    num_paths: int,
    ref_substeps: int,
) -> dict[str, float | int | str]:
    counts = np.asarray(finalized["counts"], dtype=np.float64)
    weights = counts / max(float(counts.sum()), 1.0)
    eps_alignment = np.asarray(finalized["eps_alignment"], dtype=np.float64)
    drift_alignment = np.asarray(finalized["drift_alignment"], dtype=np.float64)
    eps_delta_rms = np.asarray(finalized["eps_delta_rms"], dtype=np.float64)
    drift_delta_rms = np.asarray(finalized["drift_delta_rms"], dtype=np.float64)
    trajectory_drift = np.asarray(finalized["trajectory_drift"], dtype=np.float64)
    eps_freq_error = np.asarray(finalized["eps_freq_error"], dtype=np.float64)
    x0_freq_error = np.asarray(finalized["x0_freq_error"], dtype=np.float64)
    high_slice = high_band_slice(x0_freq_error.shape[1])
    x0_freq_total = float((weights[:, None] * x0_freq_error).sum())
    eps_freq_total = float((weights[:, None] * eps_freq_error).sum())
    x0_high = float((weights[:, None] * x0_freq_error[:, high_slice]).sum())
    eps_high = float((weights[:, None] * eps_freq_error[:, high_slice]).sum())
    third = max(1, len(trajectory_drift) // 3)
    early = slice(0, third)
    middle = slice(third, min(2 * third, len(trajectory_drift)))
    late = slice(min(2 * third, len(trajectory_drift)), len(trajectory_drift))

    return {
        "architecture": architecture,
        "solver": solver,
        "nfe": nfe,
        "eps_alignment": float((weights * eps_alignment).sum()),
        "eps_misalignment": float((weights * (1.0 - eps_alignment)).sum()),
        "drift_alignment": float((weights * drift_alignment).sum()),
        "drift_misalignment": float((weights * (1.0 - drift_alignment)).sum()),
        "eps_delta_rms": float((weights * eps_delta_rms).sum()),
        "drift_delta_rms": float((weights * drift_delta_rms).sum()),
        "trajectory_drift": float((weights * trajectory_drift).sum()),
        "endpoint_drift": float(finalized["endpoint_drift"]),
        "early_trajectory_drift": float(np.average(trajectory_drift[early], weights=np.maximum(counts[early], 1e-12))),
        "middle_trajectory_drift": float(np.average(trajectory_drift[middle], weights=np.maximum(counts[middle], 1e-12))),
        "late_trajectory_drift": float(np.average(trajectory_drift[late], weights=np.maximum(counts[late], 1e-12))),
        "x0_freq_error": x0_freq_total,
        "eps_freq_error": eps_freq_total,
        "x0_high_freq_error": x0_high,
        "eps_high_freq_error": eps_high,
        "x0_high_freq_share": x0_high / max(x0_freq_total, 1e-12),
        "eps_high_freq_share": eps_high / max(eps_freq_total, 1e-12),
        "fid": metrics.get("fid", float("nan")),
        "delta_fid": metrics.get("delta_fid", float("nan")),
        "wall_clock_sec": metrics.get("wall_clock_sec", float("nan")),
        "num_result_a_samples": int(metrics.get("num_samples", 0.0)),
        "num_paths": num_paths,
        "ref_substeps": ref_substeps,
        "checkpoint": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "noise_bank_id": noise_id,
    }


def build_bin_rows(
    finalized: dict[str, np.ndarray | float],
    architecture: str,
    solver: str,
    nfe: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    counts = np.asarray(finalized["counts"], dtype=np.float64)
    eps_alignment = np.asarray(finalized["eps_alignment"], dtype=np.float64)
    drift_alignment = np.asarray(finalized["drift_alignment"], dtype=np.float64)
    eps_delta_rms = np.asarray(finalized["eps_delta_rms"], dtype=np.float64)
    drift_delta_rms = np.asarray(finalized["drift_delta_rms"], dtype=np.float64)
    trajectory_drift = np.asarray(finalized["trajectory_drift"], dtype=np.float64)
    eps_freq_error = np.asarray(finalized["eps_freq_error"], dtype=np.float64)
    x0_freq_error = np.asarray(finalized["x0_freq_error"], dtype=np.float64)
    high_slice = high_band_slice(x0_freq_error.shape[1])
    for time_bin in range(len(counts)):
        x0_total = float(x0_freq_error[time_bin].sum())
        eps_total = float(eps_freq_error[time_bin].sum())
        rows.append(
            {
                "architecture": architecture,
                "solver": solver,
                "nfe": nfe,
                "time_bin": time_bin,
                "count": int(counts[time_bin]),
                "eps_alignment": float(eps_alignment[time_bin]),
                "eps_misalignment": float(1.0 - eps_alignment[time_bin]),
                "drift_alignment": float(drift_alignment[time_bin]),
                "drift_misalignment": float(1.0 - drift_alignment[time_bin]),
                "eps_delta_rms": float(eps_delta_rms[time_bin]),
                "drift_delta_rms": float(drift_delta_rms[time_bin]),
                "trajectory_drift": float(trajectory_drift[time_bin]),
                "x0_freq_error": x0_total,
                "eps_freq_error": eps_total,
                "x0_high_freq_share": float(x0_freq_error[time_bin, high_slice].sum() / max(x0_total, 1e-12)),
                "eps_high_freq_share": float(eps_freq_error[time_bin, high_slice].sum() / max(eps_total, 1e-12)),
            }
        )
    return rows


def build_correlation_rows(score_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    predictors = [
        "eps_misalignment",
        "drift_misalignment",
        "eps_delta_rms",
        "drift_delta_rms",
        "trajectory_drift",
        "endpoint_drift",
        "early_trajectory_drift",
        "middle_trajectory_drift",
        "late_trajectory_drift",
        "x0_freq_error",
        "eps_freq_error",
        "x0_high_freq_error",
        "eps_high_freq_error",
        "x0_high_freq_share",
        "eps_high_freq_share",
    ]
    rows = []
    groups: dict[str, list[dict[str, float | int | str]]] = {"all": score_rows}
    for architecture in sorted({str(row["architecture"]) for row in score_rows}):
        groups[architecture] = [row for row in score_rows if row["architecture"] == architecture]
    for group, group_rows in groups.items():
        y = np.asarray([float(row["delta_fid"]) for row in group_rows], dtype=np.float64)
        for predictor in predictors:
            x = np.asarray([float(row[predictor]) for row in group_rows], dtype=np.float64)
            pearson, spearman = _finite_corr_stats(x, y)
            rows.append(
                {
                    "group": group,
                    "predictor": predictor,
                    "n": int(np.isfinite(x).sum()),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                    "spearman_r": spearman,
                }
            )
    return rows


def read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_alignment_curves(output_dir: Path, score_rows: list[dict[str, float | int | str]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    architectures = sorted({str(row["architecture"]) for row in score_rows})
    fig, axes = plt.subplots(1, len(architectures), figsize=(4.7 * len(architectures), 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, architecture in zip(axes, architectures):
        for solver in SUPPORTED_SOLVERS:
            selected = sorted(
                [row for row in score_rows if row["architecture"] == architecture and row["solver"] == solver],
                key=lambda row: int(row["nfe"]),
            )
            if not selected:
                continue
            ax.plot(
                [int(row["nfe"]) for row in selected],
                [float(row["eps_misalignment"]) for row in selected],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=SOLVER_COLORS[solver],
                label=SOLVER_LABELS[solver],
            )
        ax.set_title(f"{ARCH_LABELS.get(architecture, architecture)}", fontsize=12, weight="bold")
        ax.set_xlabel("NFE")
        ax.set_ylabel("1 - epsilon direction cosine")
        ax.grid(True, alpha=0.25)
    axes[-1].legend(frameon=False)
    fig.suptitle("Figure 8. Directional Misalignment vs NFE", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure8_directional_alignment.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure8_directional_alignment.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_maps(output_dir: Path, maps: np.lib.npyio.NpzFile) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    architectures = [str(item) for item in maps["architectures"]]
    solvers = [str(item) for item in maps["solvers"]]
    nfes = [int(item) for item in maps["nfes"]]
    nfe_for_map = 20 if 20 in nfes else nfes[len(nfes) // 2]
    nfe_index = nfes.index(nfe_for_map)
    x0_freq_error = maps["x0_freq_error"]
    fig, axes = plt.subplots(len(architectures), len(solvers), figsize=(3.6 * len(solvers), 3.0 * len(architectures)))
    axes = np.atleast_2d(axes)
    last_image = None
    for ai, architecture in enumerate(architectures):
        for si, solver in enumerate(solvers):
            data = x0_freq_error[ai, si, nfe_index]
            last_image = axes[ai, si].imshow(np.log10(data + 1e-12), aspect="auto", origin="upper", cmap="viridis")
            axes[ai, si].set_title(
                f"{ARCH_LABELS.get(architecture, architecture)} {SOLVER_LABELS.get(solver, solver)}@{nfe_for_map}",
                fontsize=10,
                weight="bold",
            )
            axes[ai, si].set_xlabel("Radial frequency band")
            axes[ai, si].set_ylabel("time bin")
            axes[ai, si].tick_params(labelsize=7)
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.72, label="log10 x0-hat frequency error")
    fig.suptitle("Figure 9. Frequency-Wise Denoising Error", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure9_frequency_error_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure9_frequency_error_maps.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_drift(output_dir: Path, bin_rows: list[dict[str, float | int | str]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    architectures = sorted({str(row["architecture"]) for row in bin_rows})
    nfe_values = sorted({int(row["nfe"]) for row in bin_rows})
    nfe_for_plot = 20 if 20 in nfe_values else nfe_values[len(nfe_values) // 2]
    fig, axes = plt.subplots(1, len(architectures), figsize=(4.7 * len(architectures), 4.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, architecture in zip(axes, architectures):
        for solver in SUPPORTED_SOLVERS:
            selected = sorted(
                [
                    row
                    for row in bin_rows
                    if row["architecture"] == architecture and row["solver"] == solver and int(row["nfe"]) == nfe_for_plot
                ],
                key=lambda row: int(row["time_bin"]),
            )
            if not selected:
                continue
            ax.plot(
                [int(row["time_bin"]) for row in selected],
                [float(row["trajectory_drift"]) for row in selected],
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                color=SOLVER_COLORS[solver],
                label=SOLVER_LABELS[solver],
            )
        ax.set_title(f"{ARCH_LABELS.get(architecture, architecture)}", fontsize=12, weight="bold")
        ax.set_xlabel("time bin: noise -> data")
        ax.set_ylabel("RMS trajectory drift")
        ax.grid(True, alpha=0.25)
    axes[-1].legend(frameon=False)
    fig.suptitle(f"Figure 10. Trajectory Drift at NFE={nfe_for_plot}", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure10_trajectory_drift.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure10_trajectory_drift.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_scatter(output_dir: Path, score_rows: list[dict[str, float | int | str]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    specs = [
        ("eps_misalignment", "Directional misalignment"),
        ("x0_freq_error", "Frequency-wise x0 error"),
        ("endpoint_drift", "Endpoint trajectory drift"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.2))
    for ax, (key, title) in zip(axes, specs):
        for architecture in sorted({str(row["architecture"]) for row in score_rows}):
            selected = [row for row in score_rows if row["architecture"] == architecture]
            ax.scatter(
                [float(row[key]) for row in selected],
                [float(row["delta_fid"]) for row in selected],
                s=50,
                alpha=0.82,
                color=ARCH_COLORS.get(architecture),
                label=ARCH_LABELS.get(architecture, architecture),
            )
        x = np.asarray([float(row[key]) for row in score_rows], dtype=np.float64)
        y = np.asarray([float(row["delta_fid"]) for row in score_rows], dtype=np.float64)
        pearson, _ = _finite_corr_stats(x, y)
        if len(x) >= 2 and np.isfinite(x).all() and np.std(x) > 0:
            coef = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#111827", linewidth=1.8, alpha=0.8)
        ax.set_title(f"{title}\nPearson R2={pearson * pearson:.3f}", fontsize=12, weight="bold")
        ax.set_xlabel(key.replace("_", " "))
        ax.set_ylabel("Delta FID")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 11. P2 Trajectory Mechanisms vs Quality", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure11_p2_predictors_vs_quality.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure11_p2_predictors_vs_quality.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(output_dir: Path, score_rows: list[dict[str, float | int | str]], bin_rows: list[dict[str, float | int | str]]) -> None:
    maps = np.load(output_dir / "trajectory_maps.npz")
    plot_alignment_curves(output_dir, score_rows)
    plot_frequency_maps(output_dir, maps)
    plot_trajectory_drift(output_dir, bin_rows)
    plot_predictor_scatter(output_dir, score_rows)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.plot_only:
        score_rows = read_score_rows(output_dir / "trajectory_scores.csv")
        bin_rows = read_score_rows(output_dir / "trajectory_bins.csv")
        plot_outputs(output_dir, score_rows, bin_rows)
        print(f"Regenerated P2 trajectory figures under {output_dir / 'figures'}", flush=True)
        return

    if args.configs is None or args.run_dirs is None:
        raise ValueError("--configs and --run_dirs are required unless --plot_only is set")
    if len(args.configs) != len(args.run_dirs):
        raise ValueError("--configs and --run_dirs must have the same length")

    set_seed(args.seed)
    device = default_device()
    metrics_table = read_result_a_metrics(Path(args.metrics_csv))
    score_rows: list[dict[str, float | int | str]] = []
    bin_rows: list[dict[str, float | int | str]] = []
    architectures: list[str] = []
    all_eps_freq_maps = []
    all_x0_freq_maps = []
    all_trajectory_maps = []
    all_alignment_maps = []

    for config_path, run_dir in zip(args.configs, args.run_dirs):
        ckpt = checkpoint_path_for_run(run_dir, name=args.checkpoint_name)
        model, config, checkpoint = load_model_from_checkpoint(config_path, ckpt, device, use_ema=not args.raw_weights)
        config["training"]["batch_size"] = args.batch_size
        schedule = build_schedule(config)
        architecture = config["model"]["architecture"]
        architectures.append(architecture)
        image_size = int(config["data"].get("image_size", config["model"].get("image_size", 32)))
        noise_shape = (int(config["model"]["in_channels"]), image_size, image_size)
        noise_path = args.noise_bank or default_noise_bank_path(
            config["data"].get("root", "data"), args.seed, args.num_paths, noise_shape
        )
        noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_paths, shape=noise_shape, seed=args.seed)
        noise_id = noise_bank_id(noise_path)
        band_spec = radial_band_spec(image_size, image_size, args.freq_bands, device)
        checkpoint_id = f"step_{int(checkpoint.get('step', -1))}_images_{int(checkpoint.get('images_seen', -1))}"
        print(f"BEGIN trajectory_mechanisms architecture={architecture}", flush=True)

        arch_eps_freq_maps = []
        arch_x0_freq_maps = []
        arch_trajectory_maps = []
        arch_alignment_maps = []
        for solver in args.solvers:
            solver_eps_freq_maps = []
            solver_x0_freq_maps = []
            solver_trajectory_maps = []
            solver_alignment_maps = []
            for nfe in args.nfe:
                print(f"BEGIN trajectory architecture={architecture} solver={solver} nfe={nfe}", flush=True)
                merged = zero_feature_state(args.time_bins, args.freq_bands, device)
                for noise in noise_batches(noise_bank, args.batch_size, device):
                    batch_state = collect_solver_trajectory_features(
                        model,
                        schedule,
                        noise,
                        solver,
                        nfe,
                        args.time_bins,
                        band_spec,
                        args.ref_substeps,
                    )
                    merge_feature_state(merged, batch_state)
                finalized = finalize_state(merged)
                solver_eps_freq_maps.append(finalized["eps_freq_error"])
                solver_x0_freq_maps.append(finalized["x0_freq_error"])
                solver_trajectory_maps.append(finalized["trajectory_drift"])
                solver_alignment_maps.append(finalized["eps_alignment"])
                metrics = metrics_table.get((architecture, solver, nfe), {})
                row = summarize_features(
                    finalized,
                    architecture,
                    solver,
                    nfe,
                    metrics,
                    ckpt,
                    checkpoint_id,
                    noise_id,
                    args.num_paths,
                    args.ref_substeps,
                )
                score_rows.append(row)
                bin_rows.extend(build_bin_rows(finalized, architecture, solver, nfe))
                print(
                    f"DONE trajectory architecture={architecture} solver={solver} nfe={nfe} "
                    f"eps_misalignment={float(row['eps_misalignment']):.6e} "
                    f"x0_freq_error={float(row['x0_freq_error']):.6e} "
                    f"endpoint_drift={float(row['endpoint_drift']):.6e} "
                    f"delta_fid={metrics.get('delta_fid', float('nan')):.6f}",
                    flush=True,
                )
            arch_eps_freq_maps.append(solver_eps_freq_maps)
            arch_x0_freq_maps.append(solver_x0_freq_maps)
            arch_trajectory_maps.append(solver_trajectory_maps)
            arch_alignment_maps.append(solver_alignment_maps)
        all_eps_freq_maps.append(arch_eps_freq_maps)
        all_x0_freq_maps.append(arch_x0_freq_maps)
        all_trajectory_maps.append(arch_trajectory_maps)
        all_alignment_maps.append(arch_alignment_maps)
        print(f"DONE trajectory_mechanisms architecture={architecture}", flush=True)

    np.savez_compressed(
        output_dir / "trajectory_maps.npz",
        architectures=np.asarray(architectures),
        solvers=np.asarray(args.solvers),
        nfes=np.asarray(args.nfe),
        eps_freq_error=np.asarray(all_eps_freq_maps, dtype=np.float64),
        x0_freq_error=np.asarray(all_x0_freq_maps, dtype=np.float64),
        trajectory_drift=np.asarray(all_trajectory_maps, dtype=np.float64),
        eps_alignment=np.asarray(all_alignment_maps, dtype=np.float64),
    )
    write_csv(output_dir / "trajectory_scores.csv", score_rows)
    write_csv(output_dir / "trajectory_bins.csv", bin_rows)
    write_csv(output_dir / "trajectory_correlations.csv", build_correlation_rows(score_rows))
    plot_outputs(output_dir, score_rows, bin_rows)
    print(f"Wrote P2 trajectory mechanism outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
