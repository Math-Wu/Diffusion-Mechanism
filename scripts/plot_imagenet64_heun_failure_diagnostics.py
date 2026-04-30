from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ImageNet64 U-ViT Heun failure diagnostics.")
    parser.add_argument("--input_dir", default="outputs/imagenet64_uvit_heun_failure_diagnostics")
    parser.add_argument("--output_dir", default="outputs/imagenet64_uvit_heun_failure_diagnostics/figures")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def plot_strength_curves(metrics: list[dict[str, str]], output_dir: Path) -> None:
    p0_rows = [row for row in metrics if row["mode"] == "p0_high"]
    nfes = sorted({int(row["nfe"]) for row in p0_rows})
    strengths = sorted({float(row["strength"]) for row in p0_rows})
    lookup: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in p0_rows:
        lookup[int(row["nfe"])].append((float(row["strength"]), float(row["delta_vs_uniform"])))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for nfe in nfes:
        points = sorted(lookup[nfe])
        ax.plot([item[0] for item in points], [item[1] for item in points], marker="o", linewidth=2, label=f"NFE={nfe}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("p0_high strength")
    ax.set_ylabel("Delta FID vs uniform")
    ax.set_title("U-ViT Heun p0_high Strength Response")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "uvit_heun_strength_response.png", dpi=220)
    plt.close(fig)


def plot_density_and_gaps(density_rows: list[dict[str, str]], gap_rows: list[dict[str, str]], output_dir: Path) -> None:
    nfes = sorted({int(row["nfe"]) for row in density_rows})
    keep_strengths = {0.0, 0.1, 0.2, 0.35, 0.5}
    density_lookup: dict[tuple[int, str, float], list[tuple[int, float]]] = defaultdict(list)
    for row in density_rows:
        if row["mode"] not in {"uniform", "p0_high"}:
            continue
        strength = float(row["strength"])
        if strength not in keep_strengths:
            continue
        density_lookup[(int(row["nfe"]), row["mode"], strength)].append((int(row["bin"]), float(row["density"])))

    fig, axes = plt.subplots(1, len(nfes), figsize=(4.1 * len(nfes), 3.2), sharey=True)
    if len(nfes) == 1:
        axes = [axes]
    for ax, nfe in zip(axes, nfes):
        for mode, strength in sorted({(key[1], key[2]) for key in density_lookup if key[0] == nfe}, key=lambda x: x[1]):
            points = sorted(density_lookup[(nfe, mode, strength)])
            label = "uniform" if mode == "uniform" else f"s={strength:g}"
            ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=2 if mode == "uniform" else 1.7, label=label)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"NFE={nfe}")
        ax.set_xlabel("logSNR bin")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("sampling density")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("U-ViT Heun p0_high Time-bin Allocation")
    fig.tight_layout()
    fig.savefig(output_dir / "uvit_heun_timegrid_density.png", dpi=220)
    plt.close(fig)

    gap_lookup: dict[tuple[int, str, float], list[tuple[int, float, int]]] = defaultdict(list)
    for row in gap_rows:
        if row["mode"] not in {"uniform", "p0_high"}:
            continue
        strength = float(row["strength"])
        if strength not in {0.0, 0.35, 0.5}:
            continue
        gap_lookup[(int(row["nfe"]), row["mode"], strength)].append(
            (int(row["macro_step"]), float(row["logsnr_gap"]), int(row["full_corrector"]))
        )

    fig, axes = plt.subplots(1, len(nfes), figsize=(4.1 * len(nfes), 3.2), sharey=False)
    if len(nfes) == 1:
        axes = [axes]
    for ax, nfe in zip(axes, nfes):
        for mode, strength in sorted({(key[1], key[2]) for key in gap_lookup if key[0] == nfe}, key=lambda x: x[1]):
            points = sorted(gap_lookup[(nfe, mode, strength)])
            label = "uniform" if mode == "uniform" else f"s={strength:g}"
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                linestyle="--" if mode == "uniform" else "-",
                linewidth=2 if mode == "uniform" else 1.8,
                label=label,
            )
            for step, gap, is_full in points:
                if not is_full:
                    ax.scatter([step], [gap], s=80, facecolors="none", edgecolors="black", linewidths=1.4)
        ax.set_title(f"NFE={nfe}")
        ax.set_xlabel("macro step")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("logSNR gap")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("U-ViT Heun logSNR Gaps; hollow marker = final Euler-only interval")
    fig.tight_layout()
    fig.savefig(output_dir / "uvit_heun_timegrid_gaps.png", dpi=220)
    plt.close(fig)


def plot_p0_maps(maps_path: Path, output_dir: Path) -> None:
    data = np.load(maps_path)
    nfes = sorted(int(key.rsplit("_", 1)[-1]) for key in data.files if key.startswith("p0_high_nfe_"))
    fig, axes = plt.subplots(len(nfes), 3, figsize=(10.5, 2.6 * len(nfes)), sharex=False)
    if len(nfes) == 1:
        axes = axes[None, :]
    difficulty = data["difficulty"]
    difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
    for row_index, nfe in enumerate(nfes):
        error_map = data[f"error_map_nfe_{nfe}"]
        product = difficulty_norm * error_map
        p0_high = data[f"p0_high_nfe_{nfe}"]
        for col_index, matrix in enumerate([np.log1p(difficulty), np.log1p(error_map), np.log1p(product)]):
            ax = axes[row_index, col_index]
            im = ax.imshow(matrix.T, origin="lower", aspect="auto", cmap="magma")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_index == 0:
                ax.set_title(["log difficulty", "log local error", "log difficulty*error"][col_index])
            if col_index == 0:
                ax.set_ylabel(f"NFE={nfe}\\nfreq band")
            ax.set_xlabel("time bin")
        argmax = int(p0_high.argmax())
        for ax in axes[row_index]:
            ax.axvline(argmax, color="cyan", linewidth=1.5, linestyle="--")
    fig.suptitle("U-ViT Heun P0 Maps; cyan = p0_high argmax bin")
    fig.tight_layout()
    fig.savefig(output_dir / "uvit_heun_p0_maps.png", dpi=220)
    plt.close(fig)


def plot_bin0_ablation(metrics: list[dict[str, str]], output_dir: Path) -> None:
    nfes = sorted({int(row["nfe"]) for row in metrics})
    strengths = [0.2, 0.35, 0.5]
    fig, axes = plt.subplots(1, len(nfes), figsize=(4.2 * len(nfes), 3.6), sharey=True)
    if len(nfes) == 1:
        axes = [axes]
    for ax, nfe in zip(axes, nfes):
        groups = []
        labels = []
        for strength in strengths:
            base = [row for row in metrics if int(row["nfe"]) == nfe and row["mode"] == "p0_high" and float(row["strength"]) == strength]
            no_bin0 = [row for row in metrics if int(row["nfe"]) == nfe and row["mode"] == "p0_high_no_bin0" and float(row["strength"]) == strength]
            cap12 = [
                row
                for row in metrics
                if int(row["nfe"]) == nfe
                and row["mode"] == "p0_high_cap_bin0"
                and float(row["strength"]) == strength
                and _float_or_none(row["cap_bin0"]) == 1.2
            ]
            if base:
                groups.append(float(base[0]["delta_vs_uniform"]))
                labels.append(f"base\\n{strength:g}")
            if cap12:
                groups.append(float(cap12[0]["delta_vs_uniform"]))
                labels.append(f"cap1.2\\n{strength:g}")
            if no_bin0:
                groups.append(float(no_bin0[0]["delta_vs_uniform"]))
                labels.append(f"no0\\n{strength:g}")
        ax.bar(range(len(groups)), groups, color=["#b64b4b" if value > 0 else "#4b8f63" for value in groups])
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(groups)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"NFE={nfe}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Delta FID vs uniform")
    fig.suptitle("Bin0 Cap / Exclusion Ablation")
    fig.tight_layout()
    fig.savefig(output_dir / "uvit_heun_bin0_ablation.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_csv(input_dir / "heun_failure_metrics.csv")
    density = _read_csv(input_dir / "heun_failure_timegrid_density.csv")
    gaps = _read_csv(input_dir / "heun_failure_timegrid_gaps.csv")
    plot_strength_curves(metrics, output_dir)
    plot_density_and_gaps(density, gaps, output_dir)
    plot_p0_maps(input_dir / "heun_failure_maps.npz", output_dir)
    plot_bin0_ablation(metrics, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
