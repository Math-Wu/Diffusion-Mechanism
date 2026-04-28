from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ARCH_ORDER = ("unet", "uvit", "dit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DDIM p0_high strength ablation diagnostics.")
    parser.add_argument("--metrics_csvs", nargs="+", required=True)
    parser.add_argument("--density_csvs", nargs="+", required=True)
    parser.add_argument("--gaps_csvs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/ddim_p0_high_strength_ablation_10k_summary")
    return parser.parse_args()


def _read_csvs(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _arch_sort_key(architecture: str) -> tuple[int, str]:
    return (ARCH_ORDER.index(architecture) if architecture in ARCH_ORDER else len(ARCH_ORDER), architecture)


def _write_merged(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_delta(metrics: list[dict[str, str]], output_dir: Path) -> None:
    by_arch_strength: dict[tuple[str, float], list[tuple[int, float]]] = defaultdict(list)
    architectures = sorted({row["architecture"] for row in metrics}, key=_arch_sort_key)
    strengths = sorted({float(row["profile_strength"]) for row in metrics})
    for row in metrics:
        key = (row["architecture"], float(row["profile_strength"]))
        by_arch_strength[key].append((int(row["nfe"]), float(row["delta_vs_uniform"])))

    fig, axes = plt.subplots(1, len(architectures), figsize=(4.2 * len(architectures), 3.6), sharey=True)
    if len(architectures) == 1:
        axes = [axes]
    for axis, architecture in zip(axes, architectures):
        for strength in strengths:
            points = sorted(by_arch_strength[(architecture, strength)])
            axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker="o",
                linewidth=2,
                label=f"strength={strength:g}",
            )
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(architecture)
        axis.set_xlabel("NFE")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Delta FID vs 10k uniform DDIM")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("DDIM p0_high Strength Ablation")
    fig.tight_layout()
    fig.savefig(output_dir / "ddim_strength_delta_fid.png", dpi=220)
    plt.close(fig)


def plot_density(density_rows: list[dict[str, str]], output_dir: Path) -> None:
    architectures = sorted({row["architecture"] for row in density_rows}, key=_arch_sort_key)
    nfes = sorted({int(row["nfe"]) for row in density_rows})
    strengths = sorted({float(row["profile_strength"]) for row in density_rows})
    lookup: dict[tuple[str, int, float], list[tuple[int, float]]] = defaultdict(list)
    for row in density_rows:
        lookup[(row["architecture"], int(row["nfe"]), float(row["profile_strength"]))].append(
            (int(row["bin"]), float(row["density"]))
        )

    fig, axes = plt.subplots(len(architectures), len(nfes), figsize=(4.0 * len(nfes), 2.8 * len(architectures)), sharex=True, sharey=True)
    if len(architectures) == 1:
        axes = [axes]
    for row_index, architecture in enumerate(architectures):
        for col_index, nfe in enumerate(nfes):
            axis = axes[row_index][col_index] if len(nfes) > 1 else axes[row_index]
            for strength in strengths:
                points = sorted(lookup[(architecture, nfe, strength)])
                axis.plot(
                    [item[0] for item in points],
                    [item[1] for item in points],
                    linewidth=1.8,
                    label=f"{strength:g}",
                )
            axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
            axis.set_title(f"{architecture}, NFE={nfe}")
            axis.grid(alpha=0.2)
            if row_index == len(architectures) - 1:
                axis.set_xlabel("logSNR time-bin index")
            if col_index == 0:
                axis.set_ylabel("sampling density")
    axes[0][0].legend(title="strength", frameon=False, fontsize=8)
    fig.suptitle("DDIM p0_high Time-bin Allocation")
    fig.tight_layout()
    fig.savefig(output_dir / "ddim_timegrid_density.png", dpi=220)
    plt.close(fig)


def plot_gaps(gap_rows: list[dict[str, str]], output_dir: Path) -> None:
    architectures = sorted({row["architecture"] for row in gap_rows}, key=_arch_sort_key)
    nfes = sorted({int(row["nfe"]) for row in gap_rows})
    strengths = sorted({float(row["profile_strength"]) for row in gap_rows})
    lookup: dict[tuple[str, int, float], list[tuple[int, float]]] = defaultdict(list)
    for row in gap_rows:
        lookup[(row["architecture"], int(row["nfe"]), float(row["profile_strength"]))].append(
            (int(row["step"]), float(row["logsnr_gap"]))
        )

    fig, axes = plt.subplots(len(architectures), len(nfes), figsize=(4.0 * len(nfes), 2.8 * len(architectures)), sharey=False)
    if len(architectures) == 1:
        axes = [axes]
    for row_index, architecture in enumerate(architectures):
        for col_index, nfe in enumerate(nfes):
            axis = axes[row_index][col_index] if len(nfes) > 1 else axes[row_index]
            for strength in strengths:
                label = "uniform" if strength == 0.0 else f"{strength:g}"
                points = sorted(lookup[(architecture, nfe, strength)])
                axis.plot(
                    [item[0] for item in points],
                    [item[1] for item in points],
                    marker="o" if strength == 0.0 else None,
                    linewidth=2 if strength == 0.0 else 1.6,
                    linestyle="--" if strength == 0.0 else "-",
                    label=label,
                )
            axis.set_title(f"{architecture}, NFE={nfe}")
            axis.grid(alpha=0.2)
            if row_index == len(architectures) - 1:
                axis.set_xlabel("DDIM step")
            if col_index == 0:
                axis.set_ylabel("logSNR gap")
    axes[0][0].legend(title="grid", frameon=False, fontsize=8)
    fig.suptitle("DDIM logSNR Step Gaps")
    fig.tight_layout()
    fig.savefig(output_dir / "ddim_timegrid_gaps.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_csvs(args.metrics_csvs)
    density_rows = _read_csvs(args.density_csvs)
    gap_rows = _read_csvs(args.gaps_csvs)

    metrics = sorted(
        metrics,
        key=lambda row: (_arch_sort_key(row["architecture"]), int(row["nfe"]), float(row["profile_strength"])),
    )
    _write_merged(output_dir / "ddim_strength_metrics.csv", metrics)
    _write_merged(output_dir / "ddim_timegrid_density.csv", density_rows)
    _write_merged(output_dir / "ddim_timegrid_gaps.csv", gap_rows)

    plot_delta(metrics, output_dir)
    plot_density(density_rows, output_dir)
    plot_gaps(gap_rows, output_dir)
    print(output_dir / "ddim_strength_metrics.csv")
    print(output_dir / "ddim_strength_delta_fid.png")
    print(output_dir / "ddim_timegrid_density.png")
    print(output_dir / "ddim_timegrid_gaps.png")


if __name__ == "__main__":
    main()
