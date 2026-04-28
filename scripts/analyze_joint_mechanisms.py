from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dm_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


ARCH_LABELS = {"unet": "U-Net", "uvit": "U-ViT", "dit": "DiT"}
ARCH_COLORS = {"unet": "#0f766e", "uvit": "#9333ea", "dit": "#ea580c"}
MODEL_COLORS = {
    "p0_main": "#2563eb",
    "p2_frequency": "#f97316",
    "p2_direction": "#7e22ce",
    "p2_trajectory": "#15803d",
    "p2_core": "#db2777",
    "joint_p0_p2_frequency": "#0f766e",
    "joint_p0_p2_core": "#111827",
}


P0_FEATURES = {
    "p0_total_error": "total_error",
    "p0_difficulty_weighted_error": "difficulty_weighted_error",
    "p0_overlap_score": "overlap_score",
}
P2_FEATURES = {
    "p2_eps_misalignment": "eps_misalignment",
    "p2_drift_misalignment": "drift_misalignment",
    "p2_trajectory_drift": "trajectory_drift",
    "p2_endpoint_drift": "endpoint_drift",
    "p2_early_trajectory_drift": "early_trajectory_drift",
    "p2_late_trajectory_drift": "late_trajectory_drift",
    "p2_x0_freq_error": "x0_freq_error",
    "p2_eps_freq_error": "eps_freq_error",
    "p2_x0_high_freq_error": "x0_high_freq_error",
    "p2_eps_high_freq_error": "eps_high_freq_error",
}
LOG_FEATURES = {
    "p0_total_error",
    "p0_difficulty_weighted_error",
    "p2_eps_misalignment",
    "p2_drift_misalignment",
    "p2_trajectory_drift",
    "p2_endpoint_drift",
    "p2_early_trajectory_drift",
    "p2_late_trajectory_drift",
    "p2_x0_freq_error",
    "p2_eps_freq_error",
    "p2_x0_high_freq_error",
    "p2_eps_high_freq_error",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec("p0_main", ("p0_difficulty_weighted_error",)),
    ModelSpec("p0_all", ("p0_difficulty_weighted_error", "p0_total_error", "p0_overlap_score")),
    ModelSpec("p2_frequency", ("p2_x0_high_freq_error",)),
    ModelSpec("p2_direction", ("p2_eps_misalignment", "p2_drift_misalignment")),
    ModelSpec("p2_trajectory", ("p2_endpoint_drift", "p2_late_trajectory_drift")),
    ModelSpec(
        "p2_core",
        ("p2_x0_high_freq_error", "p2_eps_misalignment", "p2_drift_misalignment", "p2_endpoint_drift"),
    ),
    ModelSpec("joint_p0_p2_frequency", ("p0_difficulty_weighted_error", "p2_x0_high_freq_error")),
    ModelSpec(
        "joint_p0_p2_core",
        (
            "p0_difficulty_weighted_error",
            "p2_x0_high_freq_error",
            "p2_eps_misalignment",
            "p2_drift_misalignment",
            "p2_endpoint_drift",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether P2 mechanisms add explanatory power beyond P0.")
    parser.add_argument("--p0_csv", default="outputs/mechanism_p0_cifar_medium_fixedbin/mechanism_scores.csv")
    parser.add_argument("--p2_csv", default="outputs/trajectory_mechanisms_cifar_medium/trajectory_scores.csv")
    parser.add_argument("--output_dir", default="outputs/joint_mechanism_explanation")
    parser.add_argument("--target", default="delta_fid")
    parser.add_argument("--ridge_alpha", type=float, default=0.0)
    parser.add_argument("--feature_transform", choices=["log1p", "raw"], default="log1p")
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


def join_rows(p0_rows: list[dict[str, str]], p2_rows: list[dict[str, str]], target: str) -> list[dict[str, str | float | int]]:
    p2_by_key = {(row["architecture"], row["solver"], int(row["nfe"])): row for row in p2_rows}
    joined: list[dict[str, str | float | int]] = []
    missing = []
    for p0 in p0_rows:
        key = (p0["architecture"], p0["solver"], int(p0["nfe"]))
        p2 = p2_by_key.get(key)
        if p2 is None:
            missing.append(key)
            continue
        p0_target = float(p0[target])
        p2_target = float(p2[target])
        if abs(p0_target - p2_target) > 1e-6:
            raise ValueError(f"Target mismatch at {key}: P0={p0_target}, P2={p2_target}")
        row: dict[str, str | float | int] = {
            "architecture": key[0],
            "solver": key[1],
            "nfe": key[2],
            "delta_fid": p0_target,
            "fid": float(p0["fid"]),
        }
        for new_key, old_key in P0_FEATURES.items():
            row[new_key] = float(p0[old_key])
        for new_key, old_key in P2_FEATURES.items():
            row[new_key] = float(p2[old_key])
        joined.append(row)
    if missing:
        raise ValueError(f"Missing P2 rows for keys: {missing[:5]}")
    return sorted(joined, key=lambda row: (str(row["architecture"]), str(row["solver"]), int(row["nfe"])))


def feature_values(rows: list[dict[str, str | float | int]], features: tuple[str, ...], transform: str) -> np.ndarray:
    columns = []
    for feature in features:
        values = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        if transform == "log1p" and feature in LOG_FEATURES:
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        columns.append(values)
    return np.stack(columns, axis=1)


def target_values(rows: list[dict[str, str | float | int]]) -> np.ndarray:
    return np.asarray([float(row["delta_fid"]) for row in rows], dtype=np.float64)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, reference: np.ndarray | None = None) -> float:
    baseline = y_true.mean() if reference is None else reference.mean()
    denom = float(((y_true - baseline) ** 2).sum())
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / denom


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge_alpha: float = 0.0) -> np.ndarray:
    train_x, test_x = standardize(train_x, test_x)
    design = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    test_design = np.concatenate([np.ones((test_x.shape[0], 1)), test_x], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_y
    return test_design @ coef


def evaluate_spec(
    rows: list[dict[str, str | float | int]],
    spec: ModelSpec,
    ridge_alpha: float,
    group: str,
    transform: str,
) -> dict[str, str | int | float]:
    x = feature_values(rows, spec.features, transform)
    y = target_values(rows)
    in_sample = fit_predict(x, y, x, ridge_alpha=ridge_alpha)

    loo_predictions = np.zeros_like(y)
    for index in range(len(rows)):
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[index] = False
        loo_predictions[index] = fit_predict(x[train_mask], y[train_mask], x[~train_mask], ridge_alpha=ridge_alpha)[0]

    loao_predictions = np.full_like(y, np.nan)
    architectures = np.asarray([str(row["architecture"]) for row in rows])
    if group == "all" and len(set(architectures)) > 1:
        for architecture in sorted(set(architectures)):
            test_mask = architectures == architecture
            train_mask = ~test_mask
            loao_predictions[test_mask] = fit_predict(
                x[train_mask],
                y[train_mask],
                x[test_mask],
                ridge_alpha=ridge_alpha,
            )
        loao_r2 = r2_score(y, loao_predictions)
    else:
        loao_r2 = float("nan")

    return {
        "group": group,
        "model": spec.name,
        "features": " ".join(spec.features),
        "n": len(rows),
        "num_features": len(spec.features),
        "ridge_alpha": ridge_alpha,
        "feature_transform": transform,
        "in_sample_r2": r2_score(y, in_sample),
        "loo_r2": r2_score(y, loo_predictions),
        "loao_r2": loao_r2,
        "rmse": float(np.sqrt(np.mean((y - in_sample) ** 2))),
        "loo_rmse": float(np.sqrt(np.mean((y - loo_predictions) ** 2))),
    }


def build_model_comparison(
    rows: list[dict[str, str | float | int]],
    ridge_alpha: float,
    transform: str,
) -> list[dict[str, str | int | float]]:
    groups: dict[str, list[dict[str, str | float | int]]] = {"all": rows}
    for architecture in sorted({str(row["architecture"]) for row in rows}):
        groups[architecture] = [row for row in rows if row["architecture"] == architecture]
    output = []
    for group, group_rows in groups.items():
        for spec in MODEL_SPECS:
            output.append(evaluate_spec(group_rows, spec, ridge_alpha, group, transform))
    return output


def residualize(y: np.ndarray, x: np.ndarray, ridge_alpha: float) -> np.ndarray:
    predicted = fit_predict(x, y, x, ridge_alpha=ridge_alpha)
    return y - predicted


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
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson, spearman


def build_partial_correlations(
    rows: list[dict[str, str | float | int]],
    ridge_alpha: float,
    transform: str,
) -> list[dict[str, str | int | float]]:
    y = target_values(rows)
    p0_x = feature_values(rows, ("p0_difficulty_weighted_error",), transform)
    y_residual = residualize(y, p0_x, ridge_alpha)
    output = []
    for feature in P2_FEATURES:
        x = feature_values(rows, (feature,), transform)
        x_residual = residualize(x[:, 0], p0_x, ridge_alpha)
        pearson, spearman = corr_stats(x_residual, y_residual)
        output.append(
            {
                "conditioning_model": "p0_difficulty_weighted_error",
                "feature_transform": transform,
                "candidate_feature": feature,
                "n": len(rows),
                "partial_pearson_r": pearson,
                "partial_pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                "partial_spearman_r": spearman,
            }
        )
    return sorted(output, key=lambda row: -float(row["partial_pearson_r2"]))


def build_increment_rows(model_rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    by_key = {(row["group"], row["model"]): row for row in model_rows}
    pairs = [
        ("p0_main", "joint_p0_p2_frequency"),
        ("p0_main", "joint_p0_p2_core"),
        ("p2_frequency", "joint_p0_p2_frequency"),
        ("p2_core", "joint_p0_p2_core"),
    ]
    output = []
    for group in sorted({str(row["group"]) for row in model_rows}):
        for base, joint in pairs:
            base_row = by_key.get((group, base))
            joint_row = by_key.get((group, joint))
            if base_row is None or joint_row is None:
                continue
            output.append(
                {
                    "group": group,
                    "base_model": base,
                    "joint_model": joint,
                    "delta_in_sample_r2": float(joint_row["in_sample_r2"]) - float(base_row["in_sample_r2"]),
                    "delta_loo_r2": float(joint_row["loo_r2"]) - float(base_row["loo_r2"]),
                    "delta_loao_r2": float(joint_row["loao_r2"]) - float(base_row["loao_r2"])
                    if not math.isnan(float(joint_row["loao_r2"])) and not math.isnan(float(base_row["loao_r2"]))
                    else float("nan"),
                }
            )
    return output


def plot_model_comparison(output_dir: Path, rows: list[dict[str, str | int | float]]) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in rows
        if row["group"] == "all"
        and row["model"] in {"p0_main", "p2_frequency", "p2_core", "joint_p0_p2_frequency", "joint_p0_p2_core"}
    ]
    labels = [str(row["model"]) for row in selected]
    x = np.arange(len(selected))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width, [float(row["in_sample_r2"]) for row in selected], width=width, label="in-sample R2", color="#93c5fd")
    ax.bar(x, [float(row["loo_r2"]) for row in selected], width=width, label="LOO R2", color="#2563eb")
    ax.bar(x + width, [float(row["loao_r2"]) for row in selected], width=width, label="leave-architecture-out R2", color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("R2 vs Delta FID")
    ax.set_title("Figure 12. Does P2 Add Explanatory Power Beyond P0?", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure12_joint_model_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure12_joint_model_comparison.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_scatter(
    output_dir: Path,
    rows: list[dict[str, str | float | int]],
    ridge_alpha: float,
    transform: str,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    y = target_values(rows)
    p0_x = feature_values(rows, ("p0_difficulty_weighted_error",), transform)
    y_residual = residualize(y, p0_x, ridge_alpha)
    candidates = [
        ("p2_x0_high_freq_error", "Residual high-frequency x0 error"),
        ("p2_eps_misalignment", "Residual epsilon misalignment"),
        ("p2_endpoint_drift", "Residual endpoint drift"),
    ]
    fig, axes = plt.subplots(1, len(candidates), figsize=(5.0 * len(candidates), 4.2))
    for ax, (feature, title) in zip(axes, candidates):
        x_residual = residualize(feature_values(rows, (feature,), transform)[:, 0], p0_x, ridge_alpha)
        pearson, _ = corr_stats(x_residual, y_residual)
        for architecture in sorted({str(row["architecture"]) for row in rows}):
            mask = np.asarray([str(row["architecture"]) == architecture for row in rows])
            ax.scatter(
                x_residual[mask],
                y_residual[mask],
                s=50,
                alpha=0.85,
                color=ARCH_COLORS.get(architecture),
                label=ARCH_LABELS.get(architecture, architecture),
            )
        if x_residual.std() > 0:
            coef = np.polyfit(x_residual, y_residual, deg=1)
            xs = np.linspace(float(x_residual.min()), float(x_residual.max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#111827", linewidth=1.8)
        ax.set_title(f"{title}\npartial R2={pearson * pearson:.3f}", fontsize=12, weight="bold")
        ax.set_xlabel(f"{feature} residual after P0")
        ax.set_ylabel("Delta FID residual after P0")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 13. P2 Residual Signal After Removing P0", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure13_p2_residual_signal.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure13_p2_residual_signal.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(
    output_dir: Path,
    rows: list[dict[str, str | float | int]],
    ridge_alpha: float,
    transform: str,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        next(spec for spec in MODEL_SPECS if spec.name == "p0_main"),
        next(spec for spec in MODEL_SPECS if spec.name == "joint_p0_p2_frequency"),
        next(spec for spec in MODEL_SPECS if spec.name == "joint_p0_p2_core"),
    ]
    y = target_values(rows)
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.2), sharey=True)
    for ax, spec in zip(axes, specs):
        x = feature_values(rows, spec.features, transform)
        pred = fit_predict(x, y, x, ridge_alpha=ridge_alpha)
        for architecture in sorted({str(row["architecture"]) for row in rows}):
            mask = np.asarray([str(row["architecture"]) == architecture for row in rows])
            ax.scatter(
                pred[mask],
                y[mask],
                s=50,
                alpha=0.85,
                color=ARCH_COLORS.get(architecture),
                label=ARCH_LABELS.get(architecture, architecture),
            )
        lo = min(float(pred.min()), float(y.min()))
        hi = max(float(pred.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1.5, alpha=0.75)
        ax.set_title(f"{spec.name}\nin-sample R2={r2_score(y, pred):.3f}", fontsize=12, weight="bold")
        ax.set_xlabel("Predicted Delta FID")
        ax.set_ylabel("Observed Delta FID")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 14. Predicted vs Observed Quality", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "figure14_predicted_vs_observed.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure14_predicted_vs_observed.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p0_rows = read_csv(Path(args.p0_csv))
    p2_rows = read_csv(Path(args.p2_csv))
    joined = join_rows(p0_rows, p2_rows, args.target)
    write_csv(output_dir / "joint_mechanism_dataset.csv", joined)
    model_rows = build_model_comparison(joined, args.ridge_alpha, args.feature_transform)
    write_csv(output_dir / "joint_model_comparison.csv", model_rows)
    partial_rows = build_partial_correlations(joined, args.ridge_alpha, args.feature_transform)
    write_csv(output_dir / "p2_partial_correlations_after_p0.csv", partial_rows)
    increment_rows = build_increment_rows(model_rows)
    write_csv(output_dir / "joint_incremental_gains.csv", increment_rows)
    plot_model_comparison(output_dir, model_rows)
    plot_residual_scatter(output_dir, joined, args.ridge_alpha, args.feature_transform)
    plot_prediction_scatter(output_dir, joined, args.ridge_alpha, args.feature_transform)
    print(f"Wrote joint mechanism explanation outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
