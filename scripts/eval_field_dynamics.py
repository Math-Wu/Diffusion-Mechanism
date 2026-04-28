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

from dm.analysis import logsnr_bin_centers
from dm.data import build_cifar10_loaders
from dm.experiment import build_schedule, checkpoint_path_for_run, load_model_from_checkpoint
from dm.utils import default_device, ensure_dir, set_seed


ARCH_LABELS = {"unet": "U-Net", "uvit": "U-ViT", "dit": "DiT"}
ARCH_COLORS = {"unet": "#0f766e", "uvit": "#9333ea", "dit": "#ea580c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P1 field dynamics: temporal smoothness and output redundancy.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--metrics_csv", default="outputs/result_a_cifar_medium/metrics.csv")
    parser.add_argument("--output_dir", default="outputs/field_dynamics_cifar_medium")
    parser.add_argument("--checkpoint_name", default="last.pt")
    parser.add_argument("--num_data", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--time_bins", type=int, default=16)
    parser.add_argument("--nfe", nargs="+", type=int, default=[8, 15, 20, 50])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--raw_weights", action="store_true")
    return parser.parse_args()


def _batch_time(value: torch.Tensor, batch: int) -> torch.Tensor:
    return value.expand(batch)


def _logsnr_edges(schedule, bins: int, device: torch.device) -> torch.Tensor:
    t_noise = torch.tensor(1.0 - schedule.eps, device=device)
    t_data = torch.tensor(schedule.eps, device=device)
    return torch.linspace(schedule.log_snr(t_noise), schedule.log_snr(t_data), bins + 1, device=device)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).square().mean(dim=1).sqrt()


@torch.no_grad()
def compute_field_dynamics(model, schedule, val_loader, device, num_data: int, time_bins: int, seed: int) -> list[dict[str, float | int]]:
    edges = _logsnr_edges(schedule, time_bins, device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    times = logsnr_bin_centers(schedule, time_bins, device)
    half_step = 0.5 * torch.abs(edges[1] - edges[0])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sums = {
        "smoothness_same_xt": torch.zeros(time_bins, device=device),
        "curvature_same_xt": torch.zeros(time_bins, device=device),
        "redundancy_qpath": torch.zeros(time_bins, device=device),
        "qpath_delta_rms": torch.zeros(time_bins, device=device),
        "eps_rms": torch.zeros(time_bins, device=device),
    }
    counts = torch.zeros(time_bins, device=device)
    seen = 0

    for batch in val_loader:
        if seen >= num_data:
            break
        x0 = batch[0].to(device)
        if seen + x0.shape[0] > num_data:
            x0 = x0[: num_data - seen]
        seen += x0.shape[0]
        path_noise = torch.randn(x0.shape, device=device, generator=generator)

        for index, (lambda_center, t_center_scalar) in enumerate(zip(centers, times)):
            lambda_minus = torch.clamp(lambda_center - half_step, min=edges[0], max=edges[-1])
            lambda_plus = torch.clamp(lambda_center + half_step, min=edges[0], max=edges[-1])
            t_minus_scalar = schedule.inverse_log_snr(lambda_minus)
            t_plus_scalar = schedule.inverse_log_snr(lambda_plus)
            t_center = _batch_time(t_center_scalar, x0.shape[0])
            t_minus = _batch_time(t_minus_scalar, x0.shape[0])
            t_plus = _batch_time(t_plus_scalar, x0.shape[0])

            x_center = schedule.q_sample(x0, t_center, path_noise)
            eps_center = model(x_center, t_center)
            eps_minus = model(x_center, t_minus)
            eps_plus = model(x_center, t_plus)
            delta_lambda = torch.clamp(torch.abs(lambda_plus - lambda_minus), min=1e-8)
            smoothness = _rms(eps_plus - eps_minus) / delta_lambda
            curvature = _rms(eps_plus - 2.0 * eps_center + eps_minus) / delta_lambda.square()

            if index + 1 < time_bins:
                lambda_next = centers[index + 1]
            else:
                lambda_next = centers[index - 1]
            t_next_scalar = schedule.inverse_log_snr(lambda_next)
            t_next = _batch_time(t_next_scalar, x0.shape[0])
            x_next = schedule.q_sample(x0, t_next, path_noise)
            eps_next = model(x_next, t_next)
            redundancy = F.cosine_similarity(eps_center.flatten(1), eps_next.flatten(1), dim=1)
            qpath_delta = _rms(eps_next - eps_center) / torch.clamp(torch.abs(lambda_next - lambda_center), min=1e-8)

            sums["smoothness_same_xt"][index] += smoothness.sum()
            sums["curvature_same_xt"][index] += curvature.sum()
            sums["redundancy_qpath"][index] += redundancy.sum()
            sums["qpath_delta_rms"][index] += qpath_delta.sum()
            sums["eps_rms"][index] += _rms(eps_center).sum()
            counts[index] += x0.shape[0]

    if seen == 0:
        raise RuntimeError("No validation samples were available for field dynamics computation")

    rows: list[dict[str, float | int]] = []
    for index in range(time_bins):
        denom = counts[index].clamp_min(1.0)
        rows.append(
            {
                "time_bin": index,
                "t": float(times[index].detach().cpu()),
                "log_snr": float(centers[index].detach().cpu()),
                "smoothness_same_xt": float((sums["smoothness_same_xt"][index] / denom).detach().cpu()),
                "curvature_same_xt": float((sums["curvature_same_xt"][index] / denom).detach().cpu()),
                "redundancy_qpath": float((sums["redundancy_qpath"][index] / denom).detach().cpu()),
                "qpath_delta_rms": float((sums["qpath_delta_rms"][index] / denom).detach().cpu()),
                "eps_rms": float((sums["eps_rms"][index] / denom).detach().cpu()),
                "num_examples": int(counts[index].detach().cpu()),
            }
        )
    return rows


def read_result_a_metrics(path: Path) -> dict[tuple[str, str, int], float]:
    table: dict[tuple[str, str, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            solver = row["solver"]
            if solver not in {"ddim", "heun", "dpmpp", "unipc"}:
                continue
            table[(row["architecture"], solver, int(row["nfe"]))] = float(row["delta_fid"])
    return table


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr_stats(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if len(x) < 3 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def architecture_summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    summaries = []
    for architecture in sorted({str(row["architecture"]) for row in rows}):
        selected = [row for row in rows if row["architecture"] == architecture]
        early = [row for row in selected if int(row["time_bin"]) < len(selected) // 2]
        late = [row for row in selected if int(row["time_bin"]) >= len(selected) // 2]
        summary = {
            "architecture": architecture,
            "mean_smoothness": float(np.mean([float(row["smoothness_same_xt"]) for row in selected])),
            "mean_curvature": float(np.mean([float(row["curvature_same_xt"]) for row in selected])),
            "mean_redundancy": float(np.mean([float(row["redundancy_qpath"]) for row in selected])),
            "mean_qpath_delta": float(np.mean([float(row["qpath_delta_rms"]) for row in selected])),
            "early_smoothness": float(np.mean([float(row["smoothness_same_xt"]) for row in early])),
            "late_smoothness": float(np.mean([float(row["smoothness_same_xt"]) for row in late])),
            "early_redundancy": float(np.mean([float(row["redundancy_qpath"]) for row in early])),
            "late_redundancy": float(np.mean([float(row["redundancy_qpath"]) for row in late])),
        }
        summaries.append(summary)
    return summaries


def build_gain_rows(
    summaries: list[dict[str, float | str]],
    metrics: dict[tuple[str, str, int], float],
    nfes: list[int],
) -> list[dict[str, float | int | str]]:
    summary_by_arch = {str(row["architecture"]): row for row in summaries}
    rows: list[dict[str, float | int | str]] = []
    for architecture, summary in summary_by_arch.items():
        for nfe in nfes:
            if (architecture, "ddim", nfe) not in metrics:
                continue
            ddim = metrics[(architecture, "ddim", nfe)]
            heun = metrics.get((architecture, "heun", nfe), float("nan"))
            dpmpp = metrics.get((architecture, "dpmpp", nfe), float("nan"))
            unipc = metrics.get((architecture, "unipc", nfe), float("nan"))
            best_multistep = min(dpmpp, unipc)
            row = {
                "architecture": architecture,
                "nfe": nfe,
                "dpmpp_gain_vs_ddim": ddim - dpmpp,
                "unipc_gain_vs_ddim": ddim - unipc,
                "best_multistep_gain_vs_ddim": ddim - best_multistep,
                "best_multistep_gain_vs_heun": heun - best_multistep,
            }
            for key, value in summary.items():
                if key != "architecture":
                    row[key] = value
            rows.append(row)
    return rows


def build_correlation_rows(gain_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    predictors = [
        "mean_smoothness",
        "mean_curvature",
        "mean_redundancy",
        "mean_qpath_delta",
        "early_smoothness",
        "late_smoothness",
        "early_redundancy",
        "late_redundancy",
    ]
    targets = [
        "dpmpp_gain_vs_ddim",
        "unipc_gain_vs_ddim",
        "best_multistep_gain_vs_ddim",
        "best_multistep_gain_vs_heun",
    ]
    rows = []
    for target in targets:
        y = [float(row[target]) for row in gain_rows]
        for predictor in predictors:
            x = [float(row[predictor]) for row in gain_rows]
            pearson, spearman = corr_stats(x, y)
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "n": len(gain_rows),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                    "spearman_r": spearman,
                }
            )
    return rows


def plot_field_curves(output_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    metrics = [
        ("smoothness_same_xt", "Same-state temporal smoothness"),
        ("curvature_same_xt", "Same-state temporal curvature"),
        ("redundancy_qpath", "Q-path output redundancy"),
        ("qpath_delta_rms", "Q-path output delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for architecture in sorted({str(row["architecture"]) for row in rows}):
            selected = sorted([row for row in rows if row["architecture"] == architecture], key=lambda row: int(row["time_bin"]))
            ax.plot(
                [int(row["time_bin"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=ARCH_COLORS.get(architecture),
                label=ARCH_LABELS.get(architecture, architecture),
            )
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("time bin: noise -> data")
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("P1 Field Dynamics", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "field_dynamics_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "field_dynamics_curves.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_scatter(output_dir: Path, gain_rows: list[dict[str, float | int | str]]) -> None:
    figure_dir = ensure_dir(output_dir / "figures")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    specs = [
        ("mean_redundancy", "best_multistep_gain_vs_ddim", "Mean redundancy vs multistep gain"),
        ("mean_smoothness", "best_multistep_gain_vs_ddim", "Mean smoothness vs multistep gain"),
    ]
    for ax, (x_key, y_key, title) in zip(axes, specs):
        for architecture in sorted({str(row["architecture"]) for row in gain_rows}):
            selected = [row for row in gain_rows if row["architecture"] == architecture]
            ax.scatter(
                [float(row[x_key]) for row in selected],
                [float(row[y_key]) for row in selected],
                s=54,
                alpha=0.85,
                color=ARCH_COLORS.get(architecture),
                label=ARCH_LABELS.get(architecture, architecture),
            )
        x_all = [float(row[x_key]) for row in gain_rows]
        y_all = [float(row[y_key]) for row in gain_rows]
        pearson, _ = corr_stats(x_all, y_all)
        ax.set_title(f"{title}\nPearson R2={pearson * pearson:.3f}", fontsize=12, weight="bold")
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel(y_key.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("P1 Field Signature vs Solver Gain", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "field_signature_vs_solver_gain.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "field_signature_vs_solver_gain.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.configs) != len(args.run_dirs):
        raise ValueError("--configs and --run_dirs must have the same length")
    set_seed(args.seed)
    device = default_device()
    output_dir = ensure_dir(args.output_dir)
    all_rows: list[dict[str, float | int | str]] = []

    for config_path, run_dir in zip(args.configs, args.run_dirs):
        checkpoint = checkpoint_path_for_run(run_dir, name=args.checkpoint_name)
        model, config, _ = load_model_from_checkpoint(config_path, checkpoint, device, use_ema=not args.raw_weights)
        config["training"]["batch_size"] = args.batch_size
        _, val_loader = build_cifar10_loaders(config, download=False)
        schedule = build_schedule(config)
        architecture = config["model"]["architecture"]
        print(f"BEGIN field_dynamics architecture={architecture}", flush=True)
        rows = compute_field_dynamics(
            model,
            schedule,
            val_loader,
            device,
            num_data=args.num_data,
            time_bins=args.time_bins,
            seed=args.seed + 101,
        )
        for row in rows:
            row["architecture"] = architecture
            row["checkpoint"] = str(checkpoint)
        all_rows.extend(rows)
        print(f"DONE field_dynamics architecture={architecture}", flush=True)

    field_csv = output_dir / "field_dynamics.csv"
    write_csv(field_csv, all_rows)
    summaries = architecture_summary(all_rows)
    write_csv(output_dir / "field_dynamics_summary.csv", summaries)
    metrics = read_result_a_metrics(Path(args.metrics_csv))
    gain_rows = build_gain_rows(summaries, metrics, args.nfe)
    write_csv(output_dir / "solver_gain_features.csv", gain_rows)
    correlation_rows = build_correlation_rows(gain_rows)
    write_csv(output_dir / "field_gain_correlations.csv", correlation_rows)
    plot_field_curves(output_dir, all_rows)
    plot_gain_scatter(output_dir, gain_rows)
    print(f"Wrote field dynamics outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
