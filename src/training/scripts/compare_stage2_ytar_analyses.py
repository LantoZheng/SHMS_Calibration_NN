#!/usr/bin/env python3
"""Compare two Stage-2 ytar analysis output directories.

This helper reads the JSON/CSV artifacts produced by
`analyze_stage2_ytar_distribution.py`, creates side-by-side comparison figures
for the previously generated plots, and saves a metric delta table plus a
compact metric comparison chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PLOT_FILES = [
    "ytar_distribution_by_foil.png",
    "ytar_foil1_focus.png",
]

_METRIC_COLUMNS = [
    "n_events",
    "nn_sigma68",
    "nn_rmse_to_center",
    "nn_mae_to_center",
    "nn_bias_to_center",
    "root_sigma68",
    "root_rmse_to_center",
    "root_mae_to_center",
    "root_bias_to_center",
    "rmse_improvement_pct",
    "sigma68_improvement_pct",
]

_SLICE_ORDER = ["all", "foil0", "foil1", "foil2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Stage-2 ytar analysis output directories")
    parser.add_argument("--baseline-dir", required=True, help="Older analysis output directory")
    parser.add_argument("--candidate-dir", required=True, help="New analysis output directory")
    parser.add_argument("--output-dir", required=True, help="Directory to write comparison artifacts")
    return parser.parse_args()


def load_summary(directory: Path) -> dict[str, Any]:
    summary_path = directory / "ytar_distribution_summary.json"
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_metric_table(summary: dict[str, Any], run_label: str) -> pd.DataFrame:
    df = pd.DataFrame(summary["summaries"]).copy()
    df.insert(0, "run", run_label)
    return df


def save_side_by_side_plot(baseline_dir: Path, candidate_dir: Path, filename: str, output_path: Path) -> None:
    baseline_img = plt.imread(baseline_dir / filename)
    candidate_img = plt.imread(candidate_dir / filename)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for ax, img, title in [
        (axes[0], baseline_img, "Previous run"),
        (axes[1], candidate_img, "Current run"),
    ]:
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    fig.suptitle(filename.replace("_", " ").replace(".png", ""), fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_delta_table(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    left = baseline_df[["slice", *_METRIC_COLUMNS]].copy()
    right = candidate_df[["slice", *_METRIC_COLUMNS]].copy()
    merged = left.merge(right, on="slice", suffixes=("_baseline", "_candidate"))
    for col in _METRIC_COLUMNS:
        merged[f"{col}_delta"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    merged["slice"] = pd.Categorical(merged["slice"], categories=_SLICE_ORDER, ordered=True)
    merged = merged.sort_values("slice").reset_index(drop=True)
    return merged


def plot_metric_comparison(delta_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = delta_df.copy()
    slices = plot_df["slice"].astype(str).tolist()
    x = np.arange(len(slices))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        ("nn_sigma68", "NN $\\sigma_{68}$ [cm]", axes[0, 0]),
        ("nn_rmse_to_center", "NN RMSE to center [cm]", axes[0, 1]),
        ("rmse_improvement_pct", "RMSE improvement [%]", axes[1, 0]),
        ("sigma68_improvement_pct", "$\\sigma_{68}$ improvement [%]", axes[1, 1]),
    ]

    for metric, title, ax in panels:
        baseline = plot_df[f"{metric}_baseline"].to_numpy(dtype=float)
        candidate = plot_df[f"{metric}_candidate"].to_numpy(dtype=float)
        ax.bar(x - width / 2, baseline, width=width, label="Previous", color="#7f7f7f", alpha=0.8)
        ax.bar(x + width / 2, candidate, width=width, label="Current", color="#1f77b4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(slices)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()

    fig.suptitle("Stage-2 ytar analysis metric comparison", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary_payload(baseline_summary: dict[str, Any], candidate_summary: dict[str, Any], delta_df: pd.DataFrame) -> dict[str, Any]:
    focus_slice = delta_df.loc[delta_df["slice"] == "foil1"]
    focus_payload: dict[str, Any] | None = None
    if not focus_slice.empty:
        row = focus_slice.iloc[0]
        focus_payload = {
            "slice": "foil1",
            "nn_sigma68_baseline": float(row["nn_sigma68_baseline"]),
            "nn_sigma68_candidate": float(row["nn_sigma68_candidate"]),
            "nn_sigma68_delta": float(row["nn_sigma68_delta"]),
            "nn_rmse_baseline": float(row["nn_rmse_to_center_baseline"]),
            "nn_rmse_candidate": float(row["nn_rmse_to_center_candidate"]),
            "nn_rmse_delta": float(row["nn_rmse_to_center_delta"]),
            "rmse_improvement_pct_baseline": float(row["rmse_improvement_pct_baseline"]),
            "rmse_improvement_pct_candidate": float(row["rmse_improvement_pct_candidate"]),
        }
    return {
        "baseline_checkpoint": baseline_summary.get("checkpoint"),
        "candidate_checkpoint": candidate_summary.get("checkpoint"),
        "baseline_val_loss": baseline_summary.get("checkpoint_val_loss"),
        "candidate_val_loss": candidate_summary.get("checkpoint_val_loss"),
        "focus_slice": focus_payload,
        "delta_rows": delta_df.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_summary(baseline_dir)
    candidate_summary = load_summary(candidate_dir)

    baseline_df = build_metric_table(baseline_summary, "previous")
    candidate_df = build_metric_table(candidate_summary, "current")
    delta_df = build_delta_table(baseline_df, candidate_df)

    baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_metrics.csv", index=False)
    delta_df.to_csv(output_dir / "metric_deltas.csv", index=False)

    for filename in _PLOT_FILES:
        save_side_by_side_plot(
            baseline_dir,
            candidate_dir,
            filename,
            output_dir / f"compare_{filename}",
        )

    plot_metric_comparison(delta_df, output_dir / "metric_comparison.png")

    payload = build_summary_payload(baseline_summary, candidate_summary, delta_df)
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("Stage-2 ytar comparison complete")
    print(f"  baseline : {baseline_dir}")
    print(f"  current  : {candidate_dir}")
    print(f"  output   : {output_dir}")
    print("  artifacts:")
    for name in [
        "baseline_metrics.csv",
        "candidate_metrics.csv",
        "metric_deltas.csv",
        "compare_ytar_distribution_by_foil.png",
        "compare_ytar_foil1_focus.png",
        "metric_comparison.png",
        "comparison_summary.json",
    ]:
        print(f"    - {output_dir / name}")


if __name__ == "__main__":
    main()
