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
from dm.eval_utils import load_or_create_noise_bank, noise_bank_id
from dm.imagenet256 import imagenet256_schedule, load_imagenet256_model
from dm.samplers.ode import AB_COEFFS
from dm.utils import default_device, ensure_dir, set_seed

from run_imagenet256_pretrained_sweep import MODEL_SHAPES, SOLVERS, _default_noise_bank_path, _macro_intervals


MODEL_LABELS = {"adm256": "ADM-256", "uvit_l_2": "U-ViT-L/2", "dit_xl_2": "DiT-XL/2"}
MODEL_COLORS = {"adm256": "#4C78A8", "uvit_l_2": "#59A14F", "dit_xl_2": "#F28E2B"}
SOLVER_LABELS = {"ddim": "DDIM", "heun": "Heun", "dpmpp": "DPM++", "unipc": "UniPC"}
SOLVER_COLORS = {"ddim": "#4C78A8", "heun": "#F58518", "dpmpp": "#54A24B", "unipc": "#B279A2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ImageNet256 pretrained P2 trajectory mechanisms: directional alignment, "
            "frequency-wise denoising error, and trajectory drift against a high-accuracy Heun path."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_SHAPES), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--heun256_metrics")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=SOLVERS)
    parser.add_argument("--nfe", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--num_paths", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--freq_bands", type=int, default=8)
    parser.add_argument("--ref_substeps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise_bank")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_column", default="auto", choices=["auto", "delta_fid_to_heun256", "delta_fid", "fid"])
    parser.add_argument("--plot_only", action="store_true")
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).square().mean(dim=1).sqrt()


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


@torch.no_grad()
def _model_eps(model, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, amp_enabled: bool, device: torch.device) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        return model(x, t, y).float()


@torch.no_grad()
def heun_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = _model_eps(model, x, t, y, amp_enabled, device)
    drift = schedule.drift(x, t, eps)
    x_euler = x + dt * drift
    eps_next = _model_eps(model, x_euler, t_next, y, amp_enabled, device)
    drift_next = schedule.drift(x_euler, t_next, eps_next)
    return x + 0.5 * dt * (drift + drift_next)


@torch.no_grad()
def reference_heun_integrate(
    model,
    schedule,
    x: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    y: torch.Tensor,
    substeps: int,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    lambda_start = schedule.log_snr(t_start[0])
    lambda_end = schedule.log_snr(t_end[0])
    lambdas = torch.linspace(lambda_start, lambda_end, substeps + 1, device=x.device)
    times = schedule.inverse_log_snr(lambdas)
    current = x
    for index in range(substeps):
        current = heun_step(
            model,
            schedule,
            current,
            _batch_time(times[index], x.shape[0]),
            _batch_time(times[index + 1], x.shape[0]),
            y,
            amp_enabled,
            device,
        )
    return current


@torch.no_grad()
def ddim_solver_step(
    model,
    schedule,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    y: torch.Tensor,
    amp_enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    eps = _model_eps(model, x, t, y, amp_enabled, device)
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
    y: torch.Tensor,
    remaining_nfe: int,
    amp_enabled: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = _model_eps(model, x, t, y, amp_enabled, device)
    drift = schedule.drift(x, t, eps)
    remaining_nfe -= 1
    if remaining_nfe > 0:
        x_euler = x + dt * drift
        eps_next = _model_eps(model, x_euler, t_next, y, amp_enabled, device)
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
    y: torch.Tensor,
    history: list[torch.Tensor],
    max_order: int,
    amp_enabled: bool,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    dt = (t_next - t).view(x.shape[0], *([1] * (x.ndim - 1)))
    eps = _model_eps(model, x, t, y, amp_enabled, device)
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
    y: torch.Tensor,
    solver: str,
    remaining_nfe: int,
    history: list[torch.Tensor],
    amp_enabled: bool,
    device: torch.device,
) -> tuple[torch.Tensor, int, list[torch.Tensor]]:
    if solver == "ddim":
        return ddim_solver_step(model, schedule, x, t, t_next, y, amp_enabled, device), remaining_nfe - 1, history
    if solver == "heun":
        x_next, remaining_nfe = heun_solver_step(
            model, schedule, x, t, t_next, y, remaining_nfe, amp_enabled, device
        )
        return x_next, remaining_nfe, history
    if solver == "dpmpp":
        x_next, history = multistep_solver_step(
            model, schedule, x, t, t_next, y, history, max_order=2, amp_enabled=amp_enabled, device=device
        )
        return x_next, remaining_nfe - 1, history
    if solver == "unipc":
        x_next, history = multistep_solver_step(
            model, schedule, x, t, t_next, y, history, max_order=3, amp_enabled=amp_enabled, device=device
        )
        return x_next, remaining_nfe - 1, history
    raise ValueError(f"Unsupported solver: {solver}")


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


def _time_bin_index(edges: torch.Tensor, log_snr: torch.Tensor) -> int:
    value = log_snr.detach().clamp(edges[0], edges[-1])
    index = torch.searchsorted(edges, value, right=True).item() - 1
    return max(0, min(edges.numel() - 2, int(index)))


@torch.no_grad()
def collect_solver_trajectory_features(
    model,
    schedule,
    initial_noise: torch.Tensor,
    labels: torch.Tensor,
    solver: str,
    nfe: int,
    time_bins: int,
    band_spec,
    ref_substeps: int,
    amp_enabled: bool,
    device: torch.device,
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
        drift_solver = schedule.drift(x_solver, t_next, eps_solver)
        drift_reference = schedule.drift(x_reference, t_next, eps_reference)
        x0_solver = schedule.eps_to_x0(x_solver, t_next, eps_solver)
        x0_reference = schedule.eps_to_x0(x_reference, t_next, eps_reference)

        bin_index = _time_bin_index(edges, schedule.log_snr(t_next[0]))
        batch = initial_noise.shape[0]
        state["counts"][bin_index] += batch
        state["eps_alignment"][bin_index] += F.cosine_similarity(
            eps_solver.flatten(1), eps_reference.flatten(1), dim=1
        ).sum()
        state["drift_alignment"][bin_index] += F.cosine_similarity(
            drift_solver.flatten(1), drift_reference.flatten(1), dim=1
        ).sum()
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
    return {
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _heun256_reference_fid(path: str | None) -> float:
    if not path:
        return float("nan")
    rows = _read_csv_rows(Path(path))
    if not rows:
        return float("nan")
    return float(rows[0]["fid"])


def read_metrics(
    path: Path,
    model_name: str,
    heun256_fid: float,
    target_column: str,
) -> dict[tuple[str, int], dict[str, float]]:
    table: dict[tuple[str, int], dict[str, float]] = {}
    for row in _read_csv_rows(path):
        if row.get("model", row.get("architecture", "")) != model_name:
            continue
        solver = row["solver"]
        if solver not in SOLVERS:
            continue
        fid = float(row["fid"])
        if target_column == "auto":
            if "delta_fid_to_heun256" in row and row["delta_fid_to_heun256"] != "":
                delta_fid = float(row["delta_fid_to_heun256"])
            elif math.isfinite(heun256_fid):
                delta_fid = fid - heun256_fid
            elif "delta_fid" in row and row["delta_fid"] != "":
                delta_fid = float(row["delta_fid"])
            else:
                delta_fid = fid
        elif target_column == "delta_fid_to_heun256":
            delta_fid = fid - heun256_fid if math.isfinite(heun256_fid) else float(row.get("delta_fid_to_heun256", "nan"))
        elif target_column == "delta_fid":
            delta_fid = float(row["delta_fid"])
        else:
            delta_fid = fid
        table[(solver, int(row["nfe"]))] = {
            "fid": fid,
            "delta_fid": delta_fid,
            "wall_clock_sec": float(row.get("wall_clock_sec", "nan")),
            "num_samples": float(row.get("num_samples", 0.0)),
            "heun256_reference_fid": heun256_fid,
        }
    return table


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def high_band_slice(num_bands: int) -> slice:
    start = max(0, int(math.ceil(num_bands * 0.625)))
    return slice(start, num_bands)


def summarize_features(
    finalized: dict[str, np.ndarray | float],
    model_name: str,
    solver: str,
    nfe: int,
    metrics: dict[str, float],
    checkpoint: str,
    noise_id: str,
    num_paths: int,
    ref_substeps: int,
    state_space: str,
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
        "architecture": model_name,
        "model": model_name,
        "solver": solver,
        "nfe": nfe,
        "state_space": state_space,
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
        "heun256_reference_fid": metrics.get("heun256_reference_fid", float("nan")),
        "wall_clock_sec": metrics.get("wall_clock_sec", float("nan")),
        "num_result_a_samples": int(metrics.get("num_samples", 0.0)),
        "num_paths": num_paths,
        "ref_substeps": ref_substeps,
        "checkpoint": checkpoint,
        "noise_bank_id": noise_id,
    }


def build_bin_rows(
    finalized: dict[str, np.ndarray | float],
    model_name: str,
    solver: str,
    nfe: int,
    state_space: str,
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
                "architecture": model_name,
                "model": model_name,
                "solver": solver,
                "nfe": nfe,
                "state_space": state_space,
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
    for solver in sorted({str(row["solver"]) for row in score_rows}):
        groups[f"solver:{solver}"] = [row for row in score_rows if row["solver"] == solver]
    y_key = "delta_fid"
    for group, group_rows in groups.items():
        y = np.asarray([float(row[y_key]) for row in group_rows], dtype=np.float64)
        for predictor in predictors:
            x = np.asarray([float(row[predictor]) for row in group_rows], dtype=np.float64)
            pearson, spearman = _finite_corr_stats(x, y)
            rows.append(
                {
                    "group": group,
                    "predictor": predictor,
                    "target": y_key,
                    "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                    "spearman_r": spearman,
                }
            )
    return rows


def read_score_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path)


def plot_alignment_curves(output_dir: Path, score_rows: list[dict[str, float | int | str]], model_name: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for solver in SOLVERS:
        selected = sorted([row for row in score_rows if row["solver"] == solver], key=lambda row: int(row["nfe"]))
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
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)} directional misalignment")
    ax.set_xlabel("NFE")
    ax.set_ylabel("1 - epsilon direction cosine")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "figure8_directional_alignment.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure8_directional_alignment.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_maps(output_dir: Path, model_name: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    maps = np.load(output_dir / "trajectory_maps.npz")
    solvers = [str(item) for item in maps["solvers"]]
    nfes = [int(item) for item in maps["nfes"]]
    nfe_for_map = 8 if 8 in nfes else nfes[len(nfes) // 2]
    nfe_index = nfes.index(nfe_for_map)
    x0_freq_error = maps["x0_freq_error"]
    fig, axes = plt.subplots(1, len(solvers), figsize=(3.5 * len(solvers), 3.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    last_image = None
    for si, solver in enumerate(solvers):
        data = x0_freq_error[si, nfe_index]
        last_image = axes[si].imshow(np.log10(data + 1e-12), aspect="auto", origin="upper", cmap="viridis")
        axes[si].set_title(f"{SOLVER_LABELS.get(solver, solver)}@{nfe_for_map}", fontsize=10)
        axes[si].set_xlabel("radial frequency band")
        axes[si].set_ylabel("time bin" if si == 0 else "")
        axes[si].tick_params(labelsize=7)
    fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.78, label="log10 x0 frequency error")
    fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} frequency-wise denoising error")
    fig.savefig(figure_dir / "figure9_frequency_error_maps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure9_frequency_error_maps.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_drift(output_dir: Path, bin_rows: list[dict[str, float | int | str]], model_name: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    nfe_values = sorted({int(row["nfe"]) for row in bin_rows})
    nfe_for_plot = 8 if 8 in nfe_values else nfe_values[len(nfe_values) // 2]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for solver in SOLVERS:
        selected = sorted(
            [row for row in bin_rows if row["solver"] == solver and int(row["nfe"]) == nfe_for_plot],
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
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)} trajectory drift at NFE={nfe_for_plot}")
    ax.set_xlabel("time bin: noise -> data")
    ax.set_ylabel("RMS trajectory drift")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "figure10_trajectory_drift.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure10_trajectory_drift.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_scatter(output_dir: Path, score_rows: list[dict[str, float | int | str]], model_name: str) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    specs = [
        ("eps_misalignment", "Directional misalignment"),
        ("x0_high_freq_error", "High-frequency x0 error"),
        ("endpoint_drift", "Endpoint trajectory drift"),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.2), constrained_layout=True)
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
    fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} P2 mechanisms vs quality")
    fig.savefig(figure_dir / "figure11_p2_predictors_vs_quality.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure11_p2_predictors_vs_quality.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(output_dir: Path, score_rows: list[dict[str, float | int | str]], bin_rows: list[dict[str, float | int | str]], model_name: str) -> None:
    plot_alignment_curves(output_dir, score_rows, model_name)
    plot_frequency_maps(output_dir, model_name)
    plot_trajectory_drift(output_dir, bin_rows, model_name)
    plot_predictor_scatter(output_dir, score_rows, model_name)


def _load_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv_rows(path)


def _score_key(row: dict[str, str | float | int]) -> tuple[str, int]:
    return (str(row["solver"]), int(row["nfe"]))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.plot_only:
        score_rows = read_score_rows(output_dir / "trajectory_scores.csv")
        bin_rows = read_score_rows(output_dir / "trajectory_bins.csv")
        plot_outputs(output_dir, score_rows, bin_rows, args.model)
        print(f"Regenerated ImageNet256 P2 figures under {output_dir / 'figures'}", flush=True)
        return

    set_seed(args.seed)
    device = default_device()
    amp_enabled = bool(args.amp and device.type == "cuda")
    schedule = imagenet256_schedule()
    model = load_imagenet256_model(args.model, args.checkpoint, device)
    state_space = "image" if args.model == "adm256" else "latent"
    shape = MODEL_SHAPES[args.model]
    image_size = shape[-1]
    noise_path = args.noise_bank or _default_noise_bank_path(args.model, args.seed + 777001, args.num_paths, shape)
    noise_bank = load_or_create_noise_bank(noise_path, num_samples=args.num_paths, shape=shape, seed=args.seed + 777001)
    noise_id = noise_bank_id(noise_path)
    labels = (torch.arange(args.num_paths, dtype=torch.long) % 1000).contiguous()
    band_spec = radial_band_spec(image_size, image_size, args.freq_bands, device)
    heun256_fid = _heun256_reference_fid(args.heun256_metrics)
    metrics_table = read_metrics(Path(args.metrics_csv), args.model, heun256_fid, args.target_column)
    print(
        f"imagenet256_p2_start model={args.model} state_space={state_space} device={device} "
        f"num_paths={args.num_paths} batch_size={args.batch_size} amp={amp_enabled}",
        flush=True,
    )

    score_path = output_dir / "trajectory_scores.csv"
    bin_path = output_dir / "trajectory_bins.csv"
    score_rows: list[dict[str, float | int | str]] = [dict(row) for row in _load_existing(score_path)]
    bin_rows: list[dict[str, float | int | str]] = [dict(row) for row in _load_existing(bin_path)]
    completed = {_score_key(row) for row in score_rows}

    solvers = list(args.solvers)
    nfes = list(args.nfe)
    eps_freq_maps: dict[tuple[str, int], np.ndarray] = {}
    x0_freq_maps: dict[tuple[str, int], np.ndarray] = {}
    trajectory_maps: dict[tuple[str, int], np.ndarray] = {}
    alignment_maps: dict[tuple[str, int], np.ndarray] = {}
    map_path = output_dir / "trajectory_maps.npz"
    if map_path.exists():
        payload = np.load(map_path, allow_pickle=False)
        old_solvers = [str(item) for item in payload["solvers"]]
        old_nfes = [int(item) for item in payload["nfes"]]
        for si, solver in enumerate(old_solvers):
            for ni, nfe in enumerate(old_nfes):
                eps_freq_maps[(solver, nfe)] = payload["eps_freq_error"][si, ni]
                x0_freq_maps[(solver, nfe)] = payload["x0_freq_error"][si, ni]
                trajectory_maps[(solver, nfe)] = payload["trajectory_drift"][si, ni]
                alignment_maps[(solver, nfe)] = payload["eps_alignment"][si, ni]

    for solver in solvers:
        for nfe in nfes:
            key = (solver, nfe)
            if key in completed:
                print(f"skip_completed model={args.model} solver={solver} nfe={nfe}", flush=True)
                continue
            print(f"BEGIN imagenet256_p2 model={args.model} solver={solver} nfe={nfe}", flush=True)
            merged = zero_feature_state(args.time_bins, args.freq_bands, device)
            for start in range(0, noise_bank.shape[0], args.batch_size):
                end = min(start + args.batch_size, noise_bank.shape[0])
                noise = noise_bank[start:end].to(device, non_blocking=True)
                y = labels[start:end].to(device, non_blocking=True)
                batch_state = collect_solver_trajectory_features(
                    model,
                    schedule,
                    noise,
                    y,
                    solver,
                    nfe,
                    args.time_bins,
                    band_spec,
                    args.ref_substeps,
                    amp_enabled,
                    device,
                )
                merge_feature_state(merged, batch_state)
                print(
                    f"progress model={args.model} solver={solver} nfe={nfe} paths={end}/{noise_bank.shape[0]}",
                    flush=True,
                )
            finalized = finalize_state(merged)
            eps_freq_maps[key] = np.asarray(finalized["eps_freq_error"], dtype=np.float64)
            x0_freq_maps[key] = np.asarray(finalized["x0_freq_error"], dtype=np.float64)
            trajectory_maps[key] = np.asarray(finalized["trajectory_drift"], dtype=np.float64)
            alignment_maps[key] = np.asarray(finalized["eps_alignment"], dtype=np.float64)
            metrics = metrics_table.get(key, {})
            row = summarize_features(
                finalized,
                args.model,
                solver,
                nfe,
                metrics,
                args.checkpoint,
                noise_id,
                args.num_paths,
                args.ref_substeps,
                state_space,
            )
            score_rows.append(row)
            bin_rows.extend(build_bin_rows(finalized, args.model, solver, nfe, state_space))
            write_csv(score_path, score_rows)
            write_csv(bin_path, bin_rows)
            write_csv(output_dir / "trajectory_correlations.csv", build_correlation_rows(score_rows))

            eps_array = np.asarray([[eps_freq_maps.get((s, n), np.full((args.time_bins, args.freq_bands), np.nan)) for n in nfes] for s in solvers])
            x0_array = np.asarray([[x0_freq_maps.get((s, n), np.full((args.time_bins, args.freq_bands), np.nan)) for n in nfes] for s in solvers])
            trajectory_array = np.asarray([[trajectory_maps.get((s, n), np.full(args.time_bins, np.nan)) for n in nfes] for s in solvers])
            alignment_array = np.asarray([[alignment_maps.get((s, n), np.full(args.time_bins, np.nan)) for n in nfes] for s in solvers])
            np.savez_compressed(
                map_path,
                architecture=np.asarray(args.model),
                model=np.asarray(args.model),
                state_space=np.asarray(state_space),
                solvers=np.asarray(solvers),
                nfes=np.asarray(nfes),
                eps_freq_error=eps_array,
                x0_freq_error=x0_array,
                trajectory_drift=trajectory_array,
                eps_alignment=alignment_array,
            )
            print(
                f"DONE imagenet256_p2 model={args.model} solver={solver} nfe={nfe} "
                f"eps_misalignment={float(row['eps_misalignment']):.6e} "
                f"x0_high_freq_error={float(row['x0_high_freq_error']):.6e} "
                f"endpoint_drift={float(row['endpoint_drift']):.6e} "
                f"delta_fid={metrics.get('delta_fid', float('nan')):.6f}",
                flush=True,
            )

    write_csv(output_dir / "trajectory_correlations.csv", build_correlation_rows(score_rows))
    plot_outputs(output_dir, score_rows, bin_rows, args.model)
    print(f"Wrote ImageNet256 P2 trajectory mechanism outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
