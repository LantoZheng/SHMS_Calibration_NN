#!/usr/bin/env python3
"""Visualize Stage-2 labeling results for clustered sieve-hole assignments.

Outputs
-------
1. A 3x2 matrix figure:
   - left column : event-level clustering result per foil, colored by cluster
                   with cluster centers overlaid.
   - right column: cluster centers matched to final label centers, with line
                   segments showing the cluster-to-label assignment.
2. A CSV summary of cluster / label-center matching quality per foil.
3. A JSON summary listing output artifact paths.

This script is intended for HDBSCAN / two-entry labeling outputs where the
labelled CSV already contains per-event cluster assignments plus the resolved
sieve-plane label centers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_FOIL_ORDER = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage-2 labeling results")
    parser.add_argument("--data", required=True, help="Path to labelled Stage-2 CSV/Parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for output plots and summaries")
    parser.add_argument(
        "--max-points-per-foil",
        type=int,
        default=15000,
        help="Maximum event points drawn per foil for cluster scatter panels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when subsampling plotting points",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise RuntimeError("Input labelled table is missing required columns: " + ", ".join(missing))


def _label_center_columns(df: pd.DataFrame) -> tuple[str, str]:
    if {"matched_sieve_x_cm", "matched_sieve_y_cm"}.issubset(df.columns):
        return "matched_sieve_x_cm", "matched_sieve_y_cm"
    if {"candidate_sieve_x_cm", "candidate_sieve_y_cm"}.issubset(df.columns):
        return "candidate_sieve_x_cm", "candidate_sieve_y_cm"
    if {"weak_hole_xptar_center", "weak_hole_yptar_center"}.issubset(df.columns):
        tmp = df.copy()
        tmp["label_center_x_cm"] = tmp["weak_hole_xptar_center"].to_numpy(dtype=np.float64) * 253.0
        tmp["label_center_y_cm"] = tmp["weak_hole_yptar_center"].to_numpy(dtype=np.float64) * 253.0
        df["label_center_x_cm"] = tmp["label_center_x_cm"]
        df["label_center_y_cm"] = tmp["label_center_y_cm"]
        return "label_center_x_cm", "label_center_y_cm"
    raise RuntimeError("Could not resolve label-center sieve coordinates from the labelled table.")


def _subsample(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed).sort_index()


def _unique_cluster_centers(df: pd.DataFrame) -> pd.DataFrame:
    centers = (
        df[["foil_position", "cluster", "cluster_center_x", "cluster_center_y", "hole_row", "hole_col"]]
        .dropna()
        .drop_duplicates(subset=["foil_position", "cluster"])
        .sort_values(["foil_position", "cluster"])
        .reset_index(drop=True)
    )
    return centers


def _unique_label_centers(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    centers = (
        df[["foil_position", "hole_row", "hole_col", x_col, y_col]]
        .dropna()
        .drop_duplicates(subset=["foil_position", "hole_row", "hole_col"])
        .rename(columns={x_col: "label_center_x_cm", y_col: "label_center_y_cm"})
        .sort_values(["foil_position", "hole_row", "hole_col"])
        .reset_index(drop=True)
    )
    return centers


def _cluster_to_label_map(cluster_centers: pd.DataFrame, label_centers: pd.DataFrame) -> pd.DataFrame:
    merged = cluster_centers.merge(
        label_centers,
        on=["foil_position", "hole_row", "hole_col"],
        how="left",
        validate="many_to_one",
    )
    if merged[["label_center_x_cm", "label_center_y_cm"]].isna().any().any():
        missing = merged.loc[
            merged["label_center_x_cm"].isna() | merged["label_center_y_cm"].isna(),
            ["foil_position", "cluster", "hole_row", "hole_col"],
        ].head(10)
        raise RuntimeError(
            "Some clusters have no resolved label center after merge, e.g. "
            + repr(missing.to_dict(orient="records"))
        )
    merged["match_dx_cm"] = merged["cluster_center_x"] - merged["label_center_x_cm"]
    merged["match_dy_cm"] = merged["cluster_center_y"] - merged["label_center_y_cm"]
    merged["match_distance_cm"] = np.sqrt(merged["match_dx_cm"] ** 2 + merged["match_dy_cm"] ** 2)
    return merged


def _build_limits(df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = np.concatenate(
        [
            df["sieve_x"].to_numpy(dtype=np.float64),
            mapping["cluster_center_x"].to_numpy(dtype=np.float64),
            mapping["label_center_x_cm"].to_numpy(dtype=np.float64),
        ]
    )
    y_values = np.concatenate(
        [
            df["sieve_y"].to_numpy(dtype=np.float64),
            mapping["cluster_center_y"].to_numpy(dtype=np.float64),
            mapping["label_center_y_cm"].to_numpy(dtype=np.float64),
        ]
    )
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    x_lo, x_hi = np.quantile(x_values, [0.002, 0.998])
    y_lo, y_hi = np.quantile(y_values, [0.002, 0.998])
    x_pad = max((x_hi - x_lo) * 0.06, 0.8)
    y_pad = max((y_hi - y_lo) * 0.06, 0.8)
    return (float(x_lo - x_pad), float(x_hi + x_pad)), (float(y_lo - y_pad), float(y_hi + y_pad))


def _foil_summary(mapping: pd.DataFrame, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for foil, dff in mapping.groupby("foil_position", sort=True):
        distances = dff["match_distance_cm"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "foil": int(foil),
                "n_events": int((df["foil_position"] == foil).sum()),
                "n_clusters": int(dff["cluster"].nunique()),
                "n_label_centers": int(dff[["hole_row", "hole_col"]].drop_duplicates().shape[0]),
                "median_match_distance_cm": float(np.median(distances)),
                "mean_match_distance_cm": float(np.mean(distances)),
                "max_match_distance_cm": float(np.max(distances)),
                "min_match_distance_cm": float(np.min(distances)),
            }
        )
    return rows


def _plot_matrix(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    output_path: Path,
    max_points_per_foil: int,
    seed: int,
) -> None:
    foil_values = [foil for foil in _FOIL_ORDER if foil in set(df["foil_position"].dropna().astype(int).tolist())]
    if len(foil_values) != 3:
        foil_values = sorted(int(v) for v in df["foil_position"].dropna().unique())

    x_lim, y_lim = _build_limits(df, mapping)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(len(foil_values), 2, figsize=(14, 16), constrained_layout=True, sharex=True, sharey=True)
    if len(foil_values) == 1:
        axes = np.array([axes])

    for row_idx, foil in enumerate(foil_values):
        dff = df.loc[df["foil_position"] == foil].copy()
        dff_plot = _subsample(dff, max_points=max_points_per_foil, seed=seed)
        centers = mapping.loc[mapping["foil_position"] == foil].copy()

        ax_cluster = axes[row_idx, 0]
        if not dff_plot.empty:
            cluster_vals = dff_plot["cluster"].to_numpy(dtype=np.int64)
            color_vals = np.mod(cluster_vals, 256) / 255.0
            ax_cluster.scatter(
                dff_plot["sieve_x"],
                dff_plot["sieve_y"],
                c=color_vals,
                cmap=cmap,
                s=5,
                alpha=0.75,
                linewidths=0.0,
                rasterized=True,
            )
        ax_cluster.scatter(
            centers["cluster_center_x"],
            centers["cluster_center_y"],
            marker="x",
            s=48,
            linewidths=1.4,
            color="black",
            label="cluster center",
            zorder=4,
        )
        ax_cluster.set_title(
            f"foil{foil} — clustered events\n"
            f"events={len(dff):,}, clusters={centers['cluster'].nunique()}"
        )
        ax_cluster.grid(alpha=0.16)
        ax_cluster.set_aspect("equal", adjustable="box")
        ax_cluster.set_xlim(*x_lim)
        ax_cluster.set_ylim(*y_lim)
        if row_idx == len(foil_values) - 1:
            ax_cluster.set_xlabel("Sieve X [cm]")
        ax_cluster.set_ylabel("Sieve Y [cm]")
        ax_cluster.legend(loc="upper right", fontsize=8)

        ax_map = axes[row_idx, 1]
        ax_map.scatter(
            dff_plot["sieve_x"],
            dff_plot["sieve_y"],
            s=4,
            color="#c7c7c7",
            alpha=0.22,
            linewidths=0.0,
            rasterized=True,
            label="clustered events",
        )
        for row in centers.itertuples(index=False):
            ax_map.plot(
                [row.cluster_center_x, row.label_center_x_cm],
                [row.cluster_center_y, row.label_center_y_cm],
                color="#2ca02c",
                alpha=0.55,
                linewidth=0.9,
                zorder=2,
            )
        ax_map.scatter(
            centers["cluster_center_x"],
            centers["cluster_center_y"],
            marker="x",
            s=46,
            linewidths=1.3,
            color="#1f77b4",
            zorder=4,
            label="cluster center",
        )
        ax_map.scatter(
            centers["label_center_x_cm"],
            centers["label_center_y_cm"],
            marker="o",
            s=54,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.2,
            zorder=5,
            label="label center",
        )
        median_match = float(centers["match_distance_cm"].median()) if not centers.empty else float("nan")
        max_match = float(centers["match_distance_cm"].max()) if not centers.empty else float("nan")
        ax_map.set_title(
            f"foil{foil} — cluster → label centers\n"
            f"median match={median_match:.3f} cm, max={max_match:.3f} cm"
        )
        ax_map.grid(alpha=0.16)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlim(*x_lim)
        ax_map.set_ylim(*y_lim)
        if row_idx == len(foil_values) - 1:
            ax_map.set_xlabel("Sieve X [cm]")
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#c7c7c7", markeredgecolor="#c7c7c7", markersize=5, label="clustered events"),
            Line2D([0], [0], color="#2ca02c", linewidth=1.2, label="cluster→label match"),
            Line2D([0], [0], marker="x", color="#1f77b4", linestyle="None", markersize=7, markeredgewidth=1.3, label="cluster center"),
            Line2D([0], [0], marker="o", color="#d62728", markerfacecolor="none", linestyle="None", markersize=7, markeredgewidth=1.2, label="label center"),
        ]
        ax_map.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.suptitle("Stage-2 labeling results: clustering and resolved label centers", fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_table(data_path)
    _require_columns(
        df,
        [
            "foil_position",
            "cluster",
            "sieve_x",
            "sieve_y",
            "cluster_center_x",
            "cluster_center_y",
            "hole_row",
            "hole_col",
        ],
    )
    label_x_col, label_y_col = _label_center_columns(df)

    df = df.loc[df["cluster"].notna()].copy()
    df = df.loc[df["cluster"].astype(float) >= 0].copy()
    df["foil_position"] = df["foil_position"].astype(int)
    df["cluster"] = df["cluster"].astype(int)
    df["hole_row"] = df["hole_row"].astype(int)
    df["hole_col"] = df["hole_col"].astype(int)

    cluster_centers = _unique_cluster_centers(df)
    label_centers = _unique_label_centers(df, label_x_col, label_y_col)
    mapping = _cluster_to_label_map(cluster_centers, label_centers)

    matrix_plot = output_dir / "stage2_labeling_cluster_and_label_centers_3x2.png"
    summary_csv = output_dir / "stage2_labeling_match_summary.csv"
    summary_json = output_dir / "stage2_labeling_visuals_summary.json"

    _plot_matrix(
        df=df,
        mapping=mapping,
        output_path=matrix_plot,
        max_points_per_foil=int(args.max_points_per_foil),
        seed=int(args.seed),
    )

    summary_rows = _foil_summary(mapping, df)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    payload = {
        "data": str(data_path),
        "n_events": int(len(df)),
        "n_clusters_total": int(mapping[["foil_position", "cluster"]].drop_duplicates().shape[0]),
        "n_label_centers_total": int(label_centers.shape[0]),
        "artifacts": {
            "matrix_plot": str(matrix_plot),
            "summary_csv": str(summary_csv),
        },
        "summary": summary_rows,
    }
    _save_json(summary_json, payload)

    print("Stage-2 labeling visualization complete")
    print(f"  data       : {data_path}")
    print(f"  output dir : {output_dir}")
    print("  artifacts:")
    print(f"    - {matrix_plot}")
    print(f"    - {summary_csv}")
    print(f"    - {summary_json}")


if __name__ == "__main__":
    main()
