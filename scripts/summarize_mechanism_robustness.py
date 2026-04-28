from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


PREDICTORS = ("total_error", "difficulty_weighted_error", "overlap_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize robustness across mechanism P0 output directories.")
    parser.add_argument("--runs", nargs="+", required=True, help="P0 output directories containing mechanism_scores.csv.")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output_dir", default="outputs/mechanism_robustness_summary")
    return parser.parse_args()


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


def read_scores(path: Path) -> list[dict[str, str]]:
    with (path / "mechanism_scores.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(label: str, rows: list[dict[str, str]]) -> list[dict[str, str | int | float]]:
    groups: dict[str, list[dict[str, str]]] = {"all": rows}
    for architecture in sorted({row["architecture"] for row in rows}):
        groups[architecture] = [row for row in rows if row["architecture"] == architecture]
    output = []
    for group, group_rows in groups.items():
        y = [float(row["delta_fid"]) for row in group_rows]
        for predictor in PREDICTORS:
            x = [float(row[predictor]) for row in group_rows]
            pearson, spearman = corr_stats(x, y)
            output.append(
                {
                    "run": label,
                    "group": group,
                    "predictor": predictor,
                    "n": len(group_rows),
                    "pearson_r": pearson,
                    "pearson_r2": pearson * pearson if not math.isnan(pearson) else float("nan"),
                    "spearman_r": spearman,
                    "num_data": group_rows[0].get("num_data", ""),
                    "num_error": group_rows[0].get("num_noise", ""),
                    "seed": "unknown",
                }
            )
    return output


def main() -> None:
    args = parse_args()
    labels = args.labels or [Path(run).name for run in args.runs]
    if len(labels) != len(args.runs):
        raise ValueError("--labels must have the same length as --runs")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_corr_rows = []
    combined_rows = []
    for label, run in zip(labels, args.runs):
        rows = read_scores(Path(run))
        for row in rows:
            row = dict(row)
            row["run"] = label
            combined_rows.append(row)
        all_corr_rows.extend(summarize_run(label, rows))
    write_csv(output_dir / "combined_mechanism_scores.csv", combined_rows)
    write_csv(output_dir / "robustness_correlations.csv", all_corr_rows)
    print(f"Wrote robustness summary to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
