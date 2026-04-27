from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dm_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ARCH_ORDER = ("unet", "uvit", "dit")
ARCH_LABELS = {"unet": "U-Net", "uvit": "U-ViT", "dit": "DiT"}
SOLVER_ORDER = ("ddim", "heun", "dpmpp", "unipc")
SOLVER_LABELS = {"ddim": "DDIM", "heun": "Heun", "dpmpp": "DPM++", "unipc": "UniPC"}
SOLVER_COLORS = {
    "ddim": "#9a3412",
    "heun": "#2563eb",
    "dpmpp": "#15803d",
    "unipc": "#7e22ce",
}


@dataclass(frozen=True)
class MetricRow:
    architecture: str
    solver: str
    nfe: int
    fid: float
    delta_fid: float
    wall_clock_sec: float
    num_samples: int

    @property
    def ms_per_sample(self) -> float:
        return 1000.0 * self.wall_clock_sec / max(self.num_samples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CIFAR Result A quality and Pareto figures.")
    parser.add_argument("--metrics", default="outputs/result_a_cifar_medium/metrics.csv")
    parser.add_argument("--output_dir", default="outputs/result_a_cifar_medium/figures")
    parser.add_argument("--solvers", nargs="+", default=list(SOLVER_ORDER), choices=list(SOLVER_ORDER))
    parser.add_argument("--architectures", nargs="+", default=list(ARCH_ORDER), choices=list(ARCH_ORDER))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_rows(path: Path, solvers: set[str], architectures: set[str]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["solver"] not in solvers or row["architecture"] not in architectures:
                continue
            rows.append(
                MetricRow(
                    architecture=row["architecture"],
                    solver=row["solver"],
                    nfe=int(row["nfe"]),
                    fid=float(row["fid"]),
                    delta_fid=float(row["delta_fid"]),
                    wall_clock_sec=float(row["wall_clock_sec"]),
                    num_samples=int(row.get("num_samples", 10000)),
                )
            )
    if not rows:
        raise ValueError(f"No matching rows found in {path}")
    return rows


def values_for(rows: list[MetricRow], architecture: str, solver: str, field: str) -> tuple[list[int], list[float]]:
    selected = [r for r in rows if r.architecture == architecture and r.solver == solver]
    selected.sort(key=lambda r: r.nfe)
    return [r.nfe for r in selected], [float(getattr(r, field)) for r in selected]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_quality_curves(rows: list[MetricRow], architectures: list[str], solvers: list[str], output_dir: Path, dpi: int) -> None:
    specs = [
        ("delta_fid", "Delta FID vs Heun-256 reference", "figure1_quality_nfe_delta_fid", True),
        ("fid", "Proxy FID", "figure1_quality_nfe_fid_supplement", False),
    ]
    for field, ylabel, stem, use_zero_line in specs:
        fig, axes = plt.subplots(1, len(architectures), figsize=(5.1 * len(architectures), 4.0), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, architecture in zip(axes, architectures):
            for solver in solvers:
                nfes, ys = values_for(rows, architecture, solver, field)
                ax.plot(
                    nfes,
                    ys,
                    marker="o",
                    linewidth=2.2,
                    markersize=5.2,
                    color=SOLVER_COLORS[solver],
                    label=SOLVER_LABELS[solver],
                )
            if use_zero_line:
                ax.axhline(0.0, color="#262626", linewidth=1.0, linestyle="--", alpha=0.65)
                ax.set_yscale("symlog", linthresh=1.0, linscale=0.65)
            ax.set_title(ARCH_LABELS[architecture], fontsize=14, weight="bold")
            ax.set_xlabel("NFE")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", axis="both", alpha=0.25)
            ax.set_xticks(sorted({r.nfe for r in rows}))
        axes[-1].legend(frameon=False, loc="upper right")
        fig.suptitle("Result A: Quality-NFE Curves", fontsize=16, weight="bold", y=1.04)
        save_figure(fig, output_dir, stem, dpi)


def plot_best_solver_heatmap(rows: list[MetricRow], architectures: list[str], solvers: list[str], output_dir: Path, dpi: int) -> None:
    nfes = sorted({r.nfe for r in rows})
    solver_to_index = {solver: index for index, solver in enumerate(solvers)}
    values = np.full((len(architectures), len(nfes)), np.nan)
    labels: list[list[str]] = [["" for _ in nfes] for _ in architectures]
    deltas: list[list[float]] = [[float("nan") for _ in nfes] for _ in architectures]

    for ai, architecture in enumerate(architectures):
        for ni, nfe in enumerate(nfes):
            candidates = [r for r in rows if r.architecture == architecture and r.nfe == nfe and r.solver in solvers]
            best = min(candidates, key=lambda r: r.delta_fid)
            values[ai, ni] = solver_to_index[best.solver]
            labels[ai][ni] = SOLVER_LABELS[best.solver]
            deltas[ai][ni] = best.delta_fid

    cmap = ListedColormap([SOLVER_COLORS[solver] for solver in solvers])
    fig, ax = plt.subplots(figsize=(1.08 * len(nfes) + 3.0, 3.8))
    ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=len(solvers) - 0.5, aspect="auto")
    ax.set_xticks(range(len(nfes)), labels=nfes)
    ax.set_yticks(range(len(architectures)), labels=[ARCH_LABELS[a] for a in architectures])
    ax.set_xlabel("NFE")
    ax.set_title("Figure 2. Best Solver Family by Architecture and NFE", fontsize=15, weight="bold")
    for ai in range(len(architectures)):
        for ni in range(len(nfes)):
            ax.text(
                ni,
                ai,
                f"{labels[ai][ni]}\n{deltas[ai][ni]:+.2f}",
                ha="center",
                va="center",
                fontsize=8.6,
                color="white",
                weight="bold",
            )
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=SOLVER_COLORS[solver], label=SOLVER_LABELS[solver])
        for solver in solvers
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(solvers))
    save_figure(fig, output_dir, "figure2_best_solver_heatmap", dpi)


def pareto_frontier(points: list[MetricRow]) -> list[MetricRow]:
    frontier: list[MetricRow] = []
    best_delta = float("inf")
    for point in sorted(points, key=lambda r: (r.ms_per_sample, r.delta_fid)):
        if point.delta_fid < best_delta - 1e-9:
            frontier.append(point)
            best_delta = point.delta_fid
    return frontier


def plot_pareto(rows: list[MetricRow], architectures: list[str], solvers: list[str], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, len(architectures), figsize=(5.2 * len(architectures), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, architecture in zip(axes, architectures):
        arch_rows = [r for r in rows if r.architecture == architecture and r.solver in solvers]
        for solver in solvers:
            solver_rows = [r for r in arch_rows if r.solver == solver]
            ax.scatter(
                [r.ms_per_sample for r in solver_rows],
                [r.delta_fid for r in solver_rows],
                s=44,
                alpha=0.82,
                color=SOLVER_COLORS[solver],
                label=SOLVER_LABELS[solver],
            )
        frontier = pareto_frontier(arch_rows)
        ax.plot(
            [r.ms_per_sample for r in frontier],
            [r.delta_fid for r in frontier],
            color="#111827",
            linewidth=2.0,
            alpha=0.75,
        )
        best_point = min(frontier, key=lambda r: r.delta_fid)
        for point in frontier:
            should_label = point is frontier[0] or point is best_point or point.delta_fid <= 25.0
            if not should_label:
                continue
            ax.annotate(
                f"{SOLVER_LABELS[point.solver]}@{point.nfe}",
                (point.ms_per_sample, point.delta_fid),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
            )
        ax.axhline(0.0, color="#262626", linewidth=1.0, linestyle="--", alpha=0.65)
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.65)
        ax.set_xlabel("Sampling wall-clock (ms / sample)")
        ax.set_ylabel("Delta FID vs Heun-256 reference")
        ax.set_title(ARCH_LABELS[architecture], fontsize=14, weight="bold")
        ax.grid(True, which="both", alpha=0.25)
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Figure 3. Pareto Frontier", fontsize=16, weight="bold", y=1.04)
    save_figure(fig, output_dir, "figure3_pareto_frontier", dpi)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.metrics), set(args.solvers), set(args.architectures))
    plot_quality_curves(rows, args.architectures, args.solvers, output_dir, args.dpi)
    plot_best_solver_heatmap(rows, args.architectures, args.solvers, output_dir, args.dpi)
    plot_pareto(rows, args.architectures, args.solvers, output_dir, args.dpi)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
