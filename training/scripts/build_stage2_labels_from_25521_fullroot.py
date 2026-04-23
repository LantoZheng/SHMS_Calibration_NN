#!/usr/bin/env python3
"""
Build event-level Stage-2 weak labels from the 25521 full ROOT file.

Workflow
--------
1. Read the 25521 replay ROOT file with the original dotted HCANA branch names.
2. Rename columns to the underscore-based schema used by the training code.
3. Apply mild reco + PID filtering and project events to the sieve plane.
4. Classify foil positions from the P_gtr_y distribution.
5. Cluster sieve-hole patterns independently for each foil.
6. Convert cluster assignments into event-level weak labels for:
   - xptar / yptar (hole-wise angular centre + tolerance)
   - ytar        (foil-wise centre + tolerance)
7. Export the labelled event table as CSV for Stage-2 fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import SHMS_Optics_calibration_tools as soc
from SHMS_Optics_calibration_tools import DBSCANConfig, EdgeClusteringConfig, HDBSCANConfig
from physical_constraint_postprocess import merge_over_split_clusters


ROOT_BRANCHES = [
    "P.dc.x_fp",
    "P.dc.y_fp",
    "P.dc.xp_fp",
    "P.dc.yp_fp",
    "P.gtr.dp",
    "P.gtr.th",
    "P.gtr.ph",
    "P.gtr.x",
    "P.gtr.y",
    "P.react.z",
    "P.ngcer.npeSum",
    "P.hgcer.npeSum",
    "P.cal.etottracknorm",
    "P.rb.raster.fr_ybpm_tar",
    "Event_Branch/fEvtHdr/fEvtHdr.fRun",
]

RENAME_MAP = {
    "P.dc.x_fp": "P_dc_x_fp",
    "P.dc.y_fp": "P_dc_y_fp",
    "P.dc.xp_fp": "P_dc_xp_fp",
    "P.dc.yp_fp": "P_dc_yp_fp",
    "P.gtr.dp": "P_gtr_dp",
    "P.gtr.th": "P_gtr_th",
    "P.gtr.ph": "P_gtr_ph",
    "P.gtr.x": "P_gtr_x",
    "P.gtr.y": "P_gtr_y",
    "P.react.z": "P_react_z",
    "P.ngcer.npeSum": "P_ngcer_npeSum",
    "P.hgcer.npeSum": "P_hgcer_npeSum",
    "P.cal.etottracknorm": "P_cal_etottracknorm",
    "P.rb.raster.fr_ybpm_tar": "P.rb.raster.fr_ybpm_tar",
    "Event_Branch/fEvtHdr/fEvtHdr.fRun": "run_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-2 weak labels from the 25521 full ROOT replay file.")
    parser.add_argument("--root-file", required=True, help="Path to the 25521 full ROOT file")
    parser.add_argument("--tree-name", default="T", help="ROOT tree name")
    parser.add_argument(
        "--output-csv",
        default="SHMS_Calibration_NN/dataset/stage2_25521_labeled.csv",
        help="Output CSV path for Stage-2 training",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path (default: alongside output CSV)",
    )
    parser.add_argument(
        "--clustering-method",
        choices=["hdbscan", "two_entry"],
        default="hdbscan",
        help="Clustering backend used to create sieve-hole weak labels",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap for smoke tests")
    parser.add_argument("--ngcer-min", type=float, default=6.0, help="PID cut: minimum P_ngcer_npeSum")
    parser.add_argument("--hgcer-min", type=float, default=0.0, help="PID cut: minimum P_hgcer_npeSum")
    parser.add_argument("--cal-min", type=float, default=0.8, help="PID cut: minimum P_cal_etottracknorm")
    parser.add_argument("--cal-max", type=float, default=1.8, help="PID cut: maximum P_cal_etottracknorm")
    parser.add_argument("--foil-bins", type=int, default=50, help="Histogram bins for foil classification")
    parser.add_argument("--foil-sigma-factor", type=float, default=3.0, help="Foil classification window size")
    parser.add_argument("--min-xptar-tol", type=float, default=0.0015, help="Minimum xptar tolerance")
    parser.add_argument("--min-yptar-tol", type=float, default=0.0015, help="Minimum yptar tolerance")
    parser.add_argument("--min-ytar-tol", type=float, default=0.80, help="Minimum ytar tolerance")
    parser.add_argument("--cluster-merge-distance", type=float, default=0.85, help="Merge distance for over-split clusters")
    parser.add_argument("--cluster-max-size-ratio", type=float, default=3.5, help="Maximum size ratio when merging over-split clusters")
    return parser.parse_args()


def robust_half_width(series: pd.Series, minimum: float) -> float:
    values = series.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(minimum)
    q16, q84 = np.quantile(values, [0.16, 0.84])
    width = 0.5 * float(q84 - q16)
    if not np.isfinite(width):
        width = 0.0
    return float(max(width, minimum))


def load_and_standardise_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    with uproot.open(f"{args.root_file}:{args.tree_name}") as tree:
        df = tree.arrays(ROOT_BRANCHES, library="pd")

    if args.max_events is not None:
        df = df.iloc[: args.max_events].copy()

    df = df.rename(columns=RENAME_MAP)
    required_cols = list(RENAME_MAP.values())
    finite_mask = np.isfinite(df[required_cols]).all(axis=1)
    df = df.loc[finite_mask].copy()

    # Apply conservative quality cuts before sieve projection to avoid huge overflows.
    df = soc.filter_branch_ranges(
        df,
        {
            "P_gtr_dp": (-25.0, 22.0),
            "P_gtr_th": (-0.08, 0.08),
            "P_gtr_ph": (-0.06, 0.06),
            "P_react_z": (-120.0, 120.0),
        },
        verbose=False,
    )
    df = soc.add_sieve_projection(df)
    df = soc.filter_sieve_range(df, x_range=(-20.0, 20.0), y_range=(-20.0, 20.0), verbose=False)
    df = soc.filter_branch_ranges(
        df,
        {
            "P_ngcer_npeSum": (args.ngcer_min, np.inf),
            "P_hgcer_npeSum": (args.hgcer_min, np.inf),
            "P_cal_etottracknorm": (args.cal_min, args.cal_max),
        },
        verbose=False,
    )
    return df.reset_index(drop=True)


def classify_foils(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    return soc.classify_foils_with_range(
        df,
        col_name="P_gtr_y",
        bins=args.foil_bins,
        sigma_factor=args.foil_sigma_factor,
        y_range=(-5.0, 5.0),
        drop_unclassified=True,
    )


def cluster_each_foil(df: pd.DataFrame, args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    foil_positions = sorted(int(v) for v in df["foil_position"].dropna().unique() if v != -1)

    hdbscan_cfg = HDBSCANConfig(
        min_cluster_size_range=(15, 100),
        min_samples_range=(10, 50),
        target_clusters=(60, 140),
        max_cluster_size=1.8,
        distance_threshold=0.6,
        cluster_selection_method="eom",
    )
    core_cfg = DBSCANConfig(
        eps_range=(0.03, 0.30),
        target_clusters=(90, 125),
        min_samples=15,
        distance_threshold=0.8,
        max_cluster_size=2.2,
        drop_noise=True,
    )
    edge_cfg = EdgeClusteringConfig(
        radius_candidates=[-0.4, -0.2, 0.0, 0.2, 0.4],
        eps_candidates=[0.05, 0.08, 0.10, 0.12, 0.15],
        target_new_clusters=(5, 20),
        distance_threshold=0.8,
    )

    for foil_pos in foil_positions:
        df_foil = df.loc[df["foil_position"] == foil_pos].copy()
        if args.clustering_method == "hdbscan":
            df_clustered, params, n_clusters = soc.auto_hdbscan_clustering(
                df_foil,
                config=hdbscan_cfg,
                verbose=False,
            )
        else:
            df_clustered, params, n_clusters = soc.two_entry_dbscan(
                df_foil,
                core_config=core_cfg,
                edge_config=edge_cfg,
                verbose=False,
            )

        df_clustered, post_report = merge_over_split_clusters(
            df_clustered,
            merge_distance=args.cluster_merge_distance,
            max_size_ratio=args.cluster_max_size_ratio,
            verbose=False,
        )
        n_clusters_post = int(df_clustered.loc[df_clustered["cluster"] != -1, "cluster"].nunique())
        results[foil_pos] = {
            "df": df_clustered,
            "params": params,
            "n_clusters": n_clusters_post,
            "n_clusters_before_post": int(n_clusters),
            "postprocess_report": post_report,
        }
    return results


def build_event_level_labels(
    clustering_results: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labeled_frames: list[pd.DataFrame] = []
    grid_index, grid_params = soc.build_full_grid_index(clustering_results, verbose=False)
    if grid_index.empty:
        raise RuntimeError("Grid indexing returned no centres; cannot build Stage-2 weak labels.")

    grid_index = grid_index.rename(columns={"row": "hole_row", "col": "hole_col"})

    for foil_pos, result in sorted(clustering_results.items()):
        df_foil = result["df"].copy()
        df_foil = df_foil.loc[df_foil["cluster"] != -1].copy()
        if df_foil.empty:
            continue

        centers = grid_index.loc[grid_index["foil_position"] == foil_pos, ["foil_position", "cluster", "hole_row", "hole_col"]]
        df_foil = df_foil.merge(centers, on=["foil_position", "cluster"], how="inner")
        if df_foil.empty:
            continue

        cluster_stats = (
            df_foil.groupby(["foil_position", "cluster", "hole_row", "hole_col"], as_index=False)
            .agg(
                weak_hole_xptar_center=("P_gtr_th", "median"),
                weak_hole_yptar_center=("P_gtr_ph", "median"),
                hole_population=("cluster", "size"),
            )
        )
        cluster_stats["weak_hole_xptar_tol"] = (
            df_foil.groupby(["foil_position", "cluster"])["P_gtr_th"]
            .apply(lambda s: robust_half_width(s, args.min_xptar_tol))
            .to_numpy()
        )
        cluster_stats["weak_hole_yptar_tol"] = (
            df_foil.groupby(["foil_position", "cluster"])["P_gtr_ph"]
            .apply(lambda s: robust_half_width(s, args.min_yptar_tol))
            .to_numpy()
        )

        foil_stats = (
            df_foil.groupby("foil_position", as_index=False)
            .agg(
                weak_foil_ytar_center=("P_react_z", "median"),
                foil_population=("foil_position", "size"),
            )
        )
        foil_stats["weak_foil_ytar_tol"] = (
            df_foil.groupby("foil_position")["P_react_z"]
            .apply(lambda s: robust_half_width(s, args.min_ytar_tol))
            .to_numpy()
        )

        df_foil = df_foil.merge(cluster_stats, on=["foil_position", "cluster", "hole_row", "hole_col"], how="left")
        df_foil = df_foil.merge(foil_stats, on="foil_position", how="left")
        labeled_frames.append(df_foil)

    if not labeled_frames:
        raise RuntimeError("No clustered foil events survived the event-level weak-label merge.")

    labeled = pd.concat(labeled_frames, ignore_index=True)
    labeled["hole_id"] = labeled["foil_position"].astype(int) * 1000 + labeled["cluster"].astype(int)

    cluster_pop_median = max(float(labeled["hole_population"].median()), 1.0)
    labeled["weak_label_weight"] = np.clip(
        np.sqrt(labeled["hole_population"].to_numpy(dtype=np.float64) / cluster_pop_median),
        0.5,
        2.0,
    ).astype(np.float32)

    labeled["hole_row"] = labeled["hole_row"].astype(int)
    labeled["hole_col"] = labeled["hole_col"].astype(int)
    labeled["foil_position"] = labeled["foil_position"].astype(int)
    labeled["run_id"] = labeled["run_id"].astype(int)

    summary = {
        "n_labeled_events": int(len(labeled)),
        "foil_counts": {str(int(k)): int(v) for k, v in labeled["foil_position"].value_counts().sort_index().items()},
        "hole_count_total": int(labeled["hole_id"].nunique()),
        "cluster_count_per_foil": {
            str(int(foil)): int(result["n_clusters"])
            for foil, result in sorted(clustering_results.items())
        },
        "cluster_count_before_post_per_foil": {
            str(int(foil)): int(result["n_clusters_before_post"])
            for foil, result in sorted(clustering_results.items())
        },
        "grid_index_count_per_foil": {
            str(int(k)): int(v)
            for k, v in grid_index.groupby("foil_position").size().items()
        },
        "grid_params": {
            str(int(foil)): {
                "grid_spacing": float(params["grid_spacing"]),
                "row_range": [int(v) for v in params["row_range"]],
                "col_range": [int(v) for v in params["col_range"]],
                "missing_positions": len(params["missing_positions"]),
            }
            for foil, params in grid_params.items()
        },
    }
    return labeled, summary


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_name(output_csv.stem + "_summary.json")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading full ROOT from: {args.root_file}")
    df = load_and_standardise_dataframe(args)
    print(f"Events after finite/reco/PID/sieve filtering: {len(df):,}")

    df = classify_foils(df, args)
    print("Foil counts after classification:")
    print(df["foil_position"].value_counts().sort_index().to_string())

    clustering_results = cluster_each_foil(df, args)
    for foil_pos, result in sorted(clustering_results.items()):
        print(
            f"Foil {foil_pos}: clusters {result['n_clusters_before_post']} -> {result['n_clusters']} "
            f"(post), events={len(result['df']):,}"
        )

    labeled, summary = build_event_level_labels(clustering_results, args)
    labeled = labeled.sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True)

    labeled.to_csv(output_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nSaved labelled Stage-2 CSV: {output_csv}")
    print(f"Saved summary JSON       : {summary_json}")
    print(f"Labeled events           : {len(labeled):,}")
    print(f"Unique hole_id           : {labeled['hole_id'].nunique():,}")


if __name__ == "__main__":
    main()
