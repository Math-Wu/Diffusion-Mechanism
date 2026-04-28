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


ARCH_LABELS = {"unet": "U-Net", "uvit": "U-ViT", "dit": "DiT"}
SOLVER_LABELS = {"ddim": "DDIM", "heun": "Heun", "dpmpp": "DPM++", "unipc": "UniPC"}
REGION_LABELS = {
    "early": "Early",
    "middle": "Middle",
    "late": "Late",
    "low": "Low freq",
    "mid": "Mid freq",
    "high": "High freq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Band/time mediation analysis for P0 difficulty and P2 trajectory mechanisms.")
    parser.add_argument("--p0_maps", default="outputs/mechanism_p0_cifar_medium_fixedbin/mechanism_maps.npz")
    parser.add_argument("--p0_csv", default="outputs/mechanism_p0_cifar_medium_fixedbin/mechanism_scores.csv")
    parser.add_argument("--p2_maps", default="outputs/trajectory_mechanisms_cifar_medium/trajectory_maps.npz")
    parser.add_argument("--p2_csv", default="outputs/trajectory_mechanisms_cifar_medium/trajectory_scores.csv")
    parser.add_argument("--output_dir", default="outputs/band_time_mediation")
    parser.add_argument("--feature_transform", choices=["raw", "log1p"], default="log1p")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def time_slices(num_bins: int) -> dict[str, slice]:
    third = max(1, num_bins // 3)
    return {
        "early": slice(0, third),
        "middle": slice(third, min(2 * third, num_bins)),
        "late": slice(min(2 * third, num_bins), num_bins),
    }


def freq_slices(num_bands: int) -> dict[str, slice]:
    low_end = max(1, int(math.ceil(num_bands * 0.375)))
    high_start = max(low_end, int(math.ceil(num_bands * 0.625)))
    return {
        "low": slice(0, low_end),
        "mid": slice(low_end, high_start),
        "high": slice(high_start, num_bands),
    }


def corr_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x_rank = np.empty_like(x_order, dtype=np.float64)
    y_rank = np.empty_like(y_order, dtype=np.float64)
    x_rank[x_order] = np.arange(len(x), dtype=np.float64)
    y_rank[y_order] = np.arange(len(y), dtype=np.float64)
    return pearson, float(np.corrcoef(x_rank, y_rank)[0, 1])


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_x, test_x = standardize(train_x, test_x)
    design = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    test_design = np.concatenate([np.ones((test_x.shape[0], 1)), test_x], axis=1)
    coef = np.linalg.pinv(design) @ train_y
    return test_design @ coef


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(((y - y.mean()) ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - float(((y - pred) ** 2).sum()) / denom


def maybe_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    return values


def score_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    return {(row["architecture"], row["solver"], int(row["nfe"])): row for row in rows}


def build_component_rows(args: argparse.Namespace) -> list[dict[str, str | int | float]]:
    p0_maps = np.load(args.p0_maps)
    p2_maps = np.load(args.p2_maps)
    p0_rows = score_lookup(read_csv(Path(args.p0_csv)))
    p2_rows = score_lookup(read_csv(Path(args.p2_csv)))

    architectures = [str(item) for item in p0_maps["architectures"]]
    solvers = [str(item) for item in p0_maps["solvers"]]
    nfes = [int(item) for item in p0_maps["nfes"]]
    time_regions = time_slices(p0_maps["difficulty"].shape[1])
    freq_regions = freq_slices(p0_maps["difficulty"].shape[2])
    rows: list[dict[str, str | int | float]] = []

    for ai, architecture in enumerate(architectures):
        difficulty = p0_maps["difficulty"][ai]
        difficulty_norm = difficulty / max(float(difficulty.sum()), 1e-12)
        for si, solver in enumerate(solvers):
            for ni, nfe in enumerate(nfes):
                key = (architecture, solver, nfe)
                if key not in p0_rows or key not in p2_rows:
                    continue
                delta_fid = float(p0_rows[key]["delta_fid"])
                p0_error = p0_maps["solver_error"][ai, si, ni]
                p0_component = difficulty_norm * p0_error
                p2_x0 = p2_maps["x0_freq_error"][ai, si, ni]
                p2_eps = p2_maps["eps_freq_error"][ai, si, ni]
                for time_name, time_slice in time_regions.items():
                    for freq_name, freq_slice in freq_regions.items():
                        p0_value = float(p0_component[time_slice, freq_slice].sum())
                        p2_x0_value = float(p2_x0[time_slice, freq_slice].sum())
                        p2_eps_value = float(p2_eps[time_slice, freq_slice].sum())
                        rows.append(
                            {
                                "architecture": architecture,
                                "solver": solver,
                                "nfe": nfe,
                                "time_region": time_name,
                                "freq_region": freq_name,
                                "region": f"{time_name}_{freq_name}",
                                "delta_fid": delta_fid,
                                "p0_component": p0_value,
                                "p0_component_share": p0_value / max(float(p0_component.sum()), 1e-12),
                                "p2_x0_component": p2_x0_value,
                                "p2_x0_component_share": p2_x0_value / max(float(p2_x0.sum()), 1e-12),
                                "p2_eps_component": p2_eps_value,
                                "p2_eps_component_share": p2_eps_value / max(float(p2_eps.sum()), 1e-12),
                                "p0_total": float(p0_rows[key]["difficulty_weighted_error"]),
                                "p2_x0_high_total": float(p2_rows[key]["x0_high_freq_error"]),
                                "p2_eps_high_total": float(p2_rows[key]["eps_high_freq_error"]),
                            }
                        )
    return rows


def component_matrix(rows: list[dict[str, str | int | float]], key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    combo_keys = sorted({(str(row["architecture"]), str(row["solver"]), int(row["nfe"])) for row in rows})
    regions = sorted({str(row["region"]) for row in rows})
    by_key = {
        (str(row["architecture"]), str(row["solver"]), int(row["nfe"]), str(row["region"])): row
        for row in rows
    }
    matrix = np.zeros((len(combo_keys), len(regions)), dtype=np.float64)
    y = np.zeros(len(combo_keys), dtype=np.float64)
    for i, combo in enumerate(combo_keys):
        for j, region in enumerate(regions):
            row = by_key[(*combo, region)]
            matrix[i, j] = float(row[key])
            y[i] = float(row["delta_fid"])
    return regions, matrix, y


def build_component_correlations(rows: list[dict[str, str | int | float]], transform: str) -> list[dict[str, str | int | float]]:
    output = []
    groups: dict[str, list[dict[str, str | int | float]]] = {"all": rows}
    for architecture in sorted({str(row["architecture"]) for row in rows}):
        groups[architecture] = [row for row in rows if row["architecture"] == architecture]
    for group, group_rows in groups.items():
        for source_key in ["p0_component", "p2_x0_component", "p2_eps_component"]:
            regions, matrix, y = component_matrix(group_rows, source_key)
            for index, region in enumerate(regions):
                x = maybe_transform(matrix[:, index], transform)
                pearson, spearman = corr_stats(x, y)
                output.append(
                    {
                        "group": group,
                        "source": source_key,
                        "region": region,
                        "n": len(y),
                        "pearson_r": pearson,
                        "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                        "spearman_r": spearman,
                    }
                )
    return sorted(output, key=lambda row: (str(row["group"]), str(row["source"]), -float(row["pearson_r2"])))


def mediation_stats(x: np.ndarray, mediator: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = x.reshape(-1, 1)
    mediator = mediator.reshape(-1, 1)
    pred_y_x = fit_predict(x, y, x)
    pred_m_x = fit_predict(x, mediator[:, 0], x)
    xm = np.concatenate([x, mediator], axis=1)
    pred_y_xm = fit_predict(xm, y, xm)
    x_std = standardize(x, x)[0][:, 0]
    m_std = standardize(mediator, mediator)[0][:, 0]
    y_std = (y - y.mean()) / max(y.std(), 1e-12)
    a = float((np.linalg.pinv(np.c_[np.ones_like(x_std), x_std]) @ m_std)[1])
    coef_xm = np.linalg.pinv(np.c_[np.ones_like(x_std), x_std, m_std]) @ y_std
    c = float((np.linalg.pinv(np.c_[np.ones_like(x_std), x_std]) @ y_std)[1])
    b = float(coef_xm[2])
    c_prime = float(coef_xm[1])
    indirect = a * b
    return {
        "r2_x": r2_score(y, pred_y_x),
        "r2_x_mediator": r2_score(y, pred_y_xm),
        "delta_r2": r2_score(y, pred_y_xm) - r2_score(y, pred_y_x),
        "r2_mediator_from_x": r2_score(mediator[:, 0], pred_m_x),
        "standardized_a": a,
        "standardized_b": b,
        "standardized_c_total": c,
        "standardized_c_direct": c_prime,
        "standardized_indirect": indirect,
        "mediated_fraction": indirect / c if abs(c) > 1e-12 else float("nan"),
    }


def build_mediation_rows(rows: list[dict[str, str | int | float]], transform: str) -> list[dict[str, str | int | float]]:
    output = []
    p0_regions, p0_matrix, y = component_matrix(rows, "p0_component")
    p2_regions, p2_x0_matrix, _ = component_matrix(rows, "p2_x0_component")
    p2_regions_eps, p2_eps_matrix, _ = component_matrix(rows, "p2_eps_component")
    if p2_regions != p0_regions or p2_regions_eps != p0_regions:
        raise ValueError("Component region order mismatch")
    high_regions = [index for index, region in enumerate(p0_regions) if region.endswith("_high")]
    all_p0 = maybe_transform(p0_matrix.sum(axis=1), transform)
    all_x0_high = maybe_transform(p2_x0_matrix[:, high_regions].sum(axis=1), transform)
    all_eps_high = maybe_transform(p2_eps_matrix[:, high_regions].sum(axis=1), transform)
    for mediator_name, mediator in [
        ("p2_x0_high_all_time", all_x0_high),
        ("p2_eps_high_all_time", all_eps_high),
    ]:
        stats = mediation_stats(all_p0, mediator, y)
        output.append(
            {
                "x_feature": "p0_all_band_time",
                "mediator": mediator_name,
                **stats,
            }
        )

    for index, region in enumerate(p0_regions):
        p0_region = maybe_transform(p0_matrix[:, index], transform)
        p2_x0_region = maybe_transform(p2_x0_matrix[:, index], transform)
        stats = mediation_stats(p0_region, p2_x0_region, y)
        output.append({"x_feature": f"p0_{region}", "mediator": f"p2_x0_{region}", **stats})
    return sorted(output, key=lambda row: -float(row["delta_r2"]))


def plot_component_heatmaps(output_dir: Path, corr_rows: list[dict[str, str | int | float]]) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    time_names = ["early", "middle", "late"]
    freq_names = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    for ax, source in zip(axes, ["p0_component", "p2_x0_component", "p2_eps_component"]):
        data = np.zeros((len(time_names), len(freq_names)), dtype=np.float64)
        for row in corr_rows:
            if row["group"] != "all" or row["source"] != source:
                continue
            time_name, freq_name = str(row["region"]).split("_", 1)
            data[time_names.index(time_name), freq_names.index(freq_name)] = float(row["pearson_r2"])
        image = ax.imshow(data, vmin=0.0, vmax=max(1e-6, float(data.max())), aspect="auto", cmap="magma")
        ax.set_title(source.replace("_", " "), fontsize=11, weight="bold")
        ax.set_xticks(range(len(freq_names)), [REGION_LABELS[name] for name in freq_names], rotation=20, ha="right")
        ax.set_yticks(range(len(time_names)), [REGION_LABELS[name] for name in time_names])
        for i in range(len(time_names)):
            for j in range(len(freq_names)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9, weight="bold")
        fig.colorbar(image, ax=ax, shrink=0.78, label="R2 vs Delta FID")
    fig.suptitle("Figure 15. Band/Time Component Explanatory Power", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure15_band_time_component_heatmaps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure15_band_time_component_heatmaps.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mediation(output_dir: Path, rows: list[dict[str, str | int | float]]) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = rows[:10]
    labels = [str(row["x_feature"]).replace("p0_", "") for row in selected]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.bar(x, [float(row["delta_r2"]) for row in selected], color="#2563eb", alpha=0.85, label="Delta R2 from mediator")
    ax.plot(x, [float(row["r2_mediator_from_x"]) for row in selected], marker="o", color="#f97316", label="R2 mediator ~ P0 component")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Figure 16. Candidate Mediation: P0 Component -> P2 High-Frequency Error -> Delta FID", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure16_band_time_mediation.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure16_band_time_mediation.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows = build_component_rows(args)
    write_csv(output_dir / "band_time_components.csv", component_rows)
    correlation_rows = build_component_correlations(component_rows, args.feature_transform)
    write_csv(output_dir / "band_time_component_correlations.csv", correlation_rows)
    mediation_rows = build_mediation_rows(component_rows, args.feature_transform)
    write_csv(output_dir / "band_time_mediation.csv", mediation_rows)
    plot_component_heatmaps(output_dir, correlation_rows)
    plot_mediation(output_dir, mediation_rows)
    print(f"Wrote band/time mediation outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
