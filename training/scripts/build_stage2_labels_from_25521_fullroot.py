#!/usr/bin/env python3
"""
Build event-level Stage-2 labels from the 25521 full ROOT file.

Workflow
--------
1. Read the 25521 replay ROOT file with the original dotted HCANA branch names.
2. Rename columns to the underscore-based schema used by the training code.
3. Apply mild reco + PID filtering and project events to the sieve plane.
4. Classify foil positions from the P_gtr_y distribution.
5. Either cluster sieve-hole patterns independently for each foil, or directly
   assign traditional reconstructed sieve positions onto the chessboard /
   mechanical-hole grid.
6. Convert the resulting hole assignments into event-level labels for:
    - xptar / yptar (hole-wise weak labels from mechanical design parameters)
    - ytar          (foil-wise strong labels from fixed foil centres)
7. Export the labelled event table as CSV for Stage-2 fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import SHMS_Optics_calibration_tools as soc
from SHMS_Optics_calibration_tools import DBSCANConfig, EdgeClusteringConfig, HDBSCANConfig
from physical_constraint_postprocess import merge_over_split_clusters


ROOT_BRANCH_ALIASES = {
    "P_dc_x_fp": ["P_dc_x_fp", "P.dc.x_fp"],
    "P_dc_y_fp": ["P_dc_y_fp", "P.dc.y_fp"],
    "P_dc_xp_fp": ["P_dc_xp_fp", "P.dc.xp_fp"],
    "P_dc_yp_fp": ["P_dc_yp_fp", "P.dc.yp_fp"],
    "P_gtr_dp": ["P_gtr_dp", "P.gtr.dp"],
    "P_gtr_th": ["P_gtr_th", "P.gtr.th"],
    "P_gtr_ph": ["P_gtr_ph", "P.gtr.ph"],
    "P_gtr_x": ["P_gtr_x", "P.gtr.x"],
    "P_gtr_y": ["P_gtr_y", "P.gtr.y"],
    "P_react_z": ["P_react_z", "P.react.z"],
    "P_ngcer_npeSum": ["P_ngcer_npeSum", "P.ngcer.npeSum"],
    "P_hgcer_npeSum": ["P_hgcer_npeSum", "P.hgcer.npeSum"],
    "P_cal_etottracknorm": ["P_cal_etottracknorm", "P.cal.etottracknorm"],
    "P.rb.raster.fr_ybpm_tar": ["P.rb.raster.fr_ybpm_tar"],
    "P_rb_raster_fryaRawAdc": ["P_rb_raster_fryaRawAdc"],
    "P_rb_raster_frybRawAdc": ["P_rb_raster_frybRawAdc"],
    "run_id": ["run_id", "Event_Branch/fEvtHdr/fEvtHdr.fRun"],
}

REQUIRED_CANONICAL_BRANCHES = [
    "P_dc_x_fp",
    "P_dc_y_fp",
    "P_dc_xp_fp",
    "P_dc_yp_fp",
    "P_gtr_dp",
    "P_gtr_th",
    "P_gtr_ph",
    "P_gtr_x",
    "P_gtr_y",
    "P_react_z",
    "P_ngcer_npeSum",
    "P_hgcer_npeSum",
    "P_cal_etottracknorm",
]


DEFAULT_25521_FOIL_CENTERS = {
    0: 10.0,
    1: 0.0,
    2: -10.0,
}

DEFAULT_SIEVE_DISTANCE_CM = 253.0


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
        "--hole-design-table",
        default=None,
        help=(
            "Mechanical hole design table (CSV/JSON/Parquet). Required columns: "
            "foil_position, hole_row, hole_col, plus either hole_xptar_center/hole_yptar_center "
            "or weak_hole_xptar_center/weak_hole_yptar_center. Optional tolerance columns: "
            "hole_xptar_tol/hole_yptar_tol or weak_hole_xptar_tol/weak_hole_yptar_tol."
        ),
    )
    parser.add_argument(
        "--hole-x-spacing-mm",
        type=float,
        default=None,
        help=(
            "Optional sieve-hole spacing along x in mm. If provided together with --hole-y-spacing-mm, "
            "the script auto-generates the hole weak-label table from grid indices."
        ),
    )
    parser.add_argument(
        "--hole-y-spacing-mm",
        type=float,
        default=None,
        help=(
            "Optional sieve-hole spacing along y in mm. If provided together with --hole-x-spacing-mm, "
            "the script auto-generates the hole weak-label table from grid indices."
        ),
    )
    parser.add_argument(
        "--hole-tolerance-mm",
        type=float,
        default=None,
        help=(
            "Optional half-width tolerance in mm for both xptar and yptar weak labels when auto-generating "
            "the hole design table from sieve spacing."
        ),
    )
    parser.add_argument(
        "--sieve-distance-cm",
        type=float,
        default=DEFAULT_SIEVE_DISTANCE_CM,
        help=(
            "Effective target-to-sieve distance in cm used to convert hole spacing/tolerance in mm into "
            "xptar/yptar angular labels. Default: 253 cm."
        ),
    )
    parser.add_argument(
        "--hole-origin-xptar",
        type=float,
        default=0.0,
        help="Angular offset in rad applied to the auto-generated xptar hole centres.",
    )
    parser.add_argument(
        "--hole-origin-yptar",
        type=float,
        default=0.0,
        help="Angular offset in rad applied to the auto-generated yptar hole centres.",
    )
    parser.add_argument(
        "--resolved-hole-design-csv",
        default=None,
        help=(
            "Optional path to save the resolved hole design table actually used by this run. "
            "If omitted and auto-generation is used, the table is saved next to the output CSV."
        ),
    )
    parser.add_argument(
        "--foil-centers-json",
        default=None,
        help=(
            "Optional JSON file mapping foil_position -> exact ytar center in cm. "
            "If omitted, this script uses the 25521 defaults {+10, 0, -10} cm for foil {0, 1, 2}."
        ),
    )
    parser.add_argument(
        "--clustering-method",
        choices=["hdbscan", "two_entry", "direct_grid"],
        default="hdbscan",
        help="Hole-assignment backend used to create sieve-hole weak labels",
    )
    parser.add_argument(
        "--cluster-hole-assignment-mode",
        choices=["nearest", "center_out_penalized"],
        default="center_out_penalized",
        help=(
            "How clustered sieve-hole centers are matched onto the mechanical-hole grid. "
            "'nearest' uses independent nearest-neighbor matching; 'center_out_penalized' "
            "assigns inner clusters first and penalizes reusing an already-occupied hole."
        ),
    )
    parser.add_argument(
        "--cluster-hole-occupancy-penalty-cm",
        type=float,
        default=0.35,
        help=(
            "Penalty in cm added to a candidate hole's effective match cost for each cluster "
            "already assigned to that hole when using center_out_penalized matching."
        ),
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
    parser.add_argument(
        "--direct-grid-min-hole-population",
        type=int,
        default=15,
        help="Minimum events required for an occupied chessboard / mechanical hole to be kept in direct_grid mode.",
    )
    parser.add_argument(
        "--direct-grid-max-match-distance-cm",
        type=float,
        default=None,
        help=(
            "Maximum allowed distance in cm between a reconstructed sieve point and its assigned chessboard / "
            "mechanical-hole centre in direct_grid mode. Default: half of the grid-cell diagonal when spacing is known."
        ),
    )
    parser.add_argument(
        "--direct-grid-max-holes-per-foil",
        type=int,
        default=None,
        help="Optional maximum number of occupied holes kept per foil in direct_grid mode, ranked by population then match quality.",
    )
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


def _load_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".json":
        with open(table_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "rows" in payload:
            return pd.DataFrame(payload["rows"])
        if isinstance(payload, dict):
            return pd.DataFrame(payload)
    raise ValueError(f"Unsupported table format for: {path}")


def mm_to_target_angle(mm_value: float, sieve_distance_cm: float) -> float:
    sieve_distance_cm = float(sieve_distance_cm)
    if sieve_distance_cm <= 0:
        raise ValueError(f"sieve_distance_cm must be positive, got {sieve_distance_cm}")
    return float((float(mm_value) / 10.0) / sieve_distance_cm)


def target_angle_to_sieve_cm(angle_value: float, sieve_distance_cm: float) -> float:
    return float(angle_value) * float(sieve_distance_cm)


def infer_run_id(root_file: str) -> int:
    matches = re.findall(r"(\d{4,})", str(root_file))
    if matches:
        return int(matches[-1])
    return -1


def resolve_root_branch_map(tree: Any) -> dict[str, str]:
    tree_keys = set(cast(list[str], tree.keys()))
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical_name, candidates in ROOT_BRANCH_ALIASES.items():
        found = next((candidate for candidate in candidates if candidate in tree_keys), None)
        if found is not None:
            resolved[found] = canonical_name
        elif canonical_name in REQUIRED_CANONICAL_BRANCHES:
            missing.append(canonical_name)

    if missing:
        raise KeyError(
            "ROOT file is missing required branches for Stage-2 label building: " + ", ".join(missing)
        )

    return resolved


def extract_cluster_centers(clustering_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for foil_pos, result in sorted(clustering_results.items()):
        df_foil = result["df"].copy()
        df_foil = df_foil.loc[df_foil["cluster"] != -1].copy()
        if df_foil.empty:
            continue
        centers = (
            df_foil.groupby("cluster", as_index=False)
            .agg(
                cluster_center_x=("cluster_center_x", "median"),
                cluster_center_y=("cluster_center_y", "median"),
            )
        )
        centers["foil_position"] = int(foil_pos)
        frames.append(centers[["foil_position", "cluster", "cluster_center_x", "cluster_center_y"]])

    if not frames:
        return pd.DataFrame(columns=["foil_position", "cluster", "cluster_center_x", "cluster_center_y"])
    return pd.concat(frames, ignore_index=True)


def infer_lattice_origin_cm(values_cm: pd.Series, spacing_cm: float, initial_origin_cm: float = 0.0) -> float:
    values = values_cm.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(initial_origin_cm)
    nearest_index = np.rint((values - float(initial_origin_cm)) / float(spacing_cm))
    residual = values - (float(initial_origin_cm) + nearest_index * float(spacing_cm))
    return float(initial_origin_cm + np.median(residual))


def load_mechanical_hole_design(args: argparse.Namespace) -> pd.DataFrame:
    if not args.hole_design_table:
        raise ValueError(
            "Mechanical hole design table is required unless --hole-x-spacing-mm and --hole-y-spacing-mm are provided."
        )

    design = _load_table(args.hole_design_table).copy()
    rename_map = {
        "xptar_center": "weak_hole_xptar_center",
        "yptar_center": "weak_hole_yptar_center",
        "xptar_tol": "weak_hole_xptar_tol",
        "yptar_tol": "weak_hole_yptar_tol",
        "hole_xptar_center": "weak_hole_xptar_center",
        "hole_yptar_center": "weak_hole_yptar_center",
        "hole_xptar_tol": "weak_hole_xptar_tol",
        "hole_yptar_tol": "weak_hole_yptar_tol",
    }
    design = design.rename(columns=rename_map)

    required = ["foil_position", "hole_row", "hole_col", "weak_hole_xptar_center", "weak_hole_yptar_center"]
    missing = [col for col in required if col not in design.columns]
    if missing:
        raise ValueError(
            "Mechanical hole design table is missing required columns: " + ", ".join(missing)
        )

    if "weak_hole_xptar_tol" not in design.columns:
        design["weak_hole_xptar_tol"] = float(args.min_xptar_tol)
    if "weak_hole_yptar_tol" not in design.columns:
        design["weak_hole_yptar_tol"] = float(args.min_yptar_tol)

    design = design[
        [
            "foil_position",
            "hole_row",
            "hole_col",
            "weak_hole_xptar_center",
            "weak_hole_yptar_center",
            "weak_hole_xptar_tol",
            "weak_hole_yptar_tol",
        ]
    ].copy()
    design["foil_position"] = design["foil_position"].astype(int)
    design["hole_row"] = design["hole_row"].astype(int)
    design["hole_col"] = design["hole_col"].astype(int)

    duplicated = design.duplicated(subset=["foil_position", "hole_row", "hole_col"], keep=False)
    if duplicated.any():
        dup_rows = design.loc[duplicated, ["foil_position", "hole_row", "hole_col"]]
        raise ValueError(
            "Mechanical hole design table contains duplicate (foil_position, hole_row, hole_col) entries: "
            f"{dup_rows.head(10).to_dict(orient='records')}"
        )

    return design


def ensure_candidate_sieve_columns(
    hole_design: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    candidate_design = hole_design.copy()
    if "candidate_sieve_x_cm" not in candidate_design.columns:
        candidate_design["candidate_sieve_x_cm"] = (
            candidate_design["weak_hole_xptar_center"].to_numpy(dtype=np.float64) * float(args.sieve_distance_cm)
        )
    if "candidate_sieve_y_cm" not in candidate_design.columns:
        candidate_design["candidate_sieve_y_cm"] = (
            candidate_design["weak_hole_yptar_center"].to_numpy(dtype=np.float64) * float(args.sieve_distance_cm)
        )
    return candidate_design


def build_observed_mechanical_hole_design_from_event_grid(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_spacing_mm = getattr(args, "hole_x_spacing_mm", None)
    y_spacing_mm = getattr(args, "hole_y_spacing_mm", None)
    tolerance_mm = getattr(args, "hole_tolerance_mm", None)
    if x_spacing_mm is None or y_spacing_mm is None:
        raise ValueError(
            "direct_grid auto-generation requires both --hole-x-spacing-mm and --hole-y-spacing-mm."
        )
    if tolerance_mm is None:
        raise ValueError("direct_grid auto-generation requires --hole-tolerance-mm.")
    if df.empty:
        raise RuntimeError("No events available; cannot infer chessboard / mechanical-hole grid.")

    x_spacing_cm = float(x_spacing_mm) / 10.0
    y_spacing_cm = float(y_spacing_mm) / 10.0
    tolerance_rad = mm_to_target_angle(float(tolerance_mm), float(args.sieve_distance_cm))
    initial_x_origin_cm = target_angle_to_sieve_cm(float(args.hole_origin_xptar), float(args.sieve_distance_cm))
    initial_y_origin_cm = target_angle_to_sieve_cm(float(args.hole_origin_yptar), float(args.sieve_distance_cm))

    x_origin_cm = infer_lattice_origin_cm(df["sieve_x"], x_spacing_cm, initial_x_origin_cm)
    y_origin_cm = infer_lattice_origin_cm(df["sieve_y"], y_spacing_cm, initial_y_origin_cm)

    grid = df[["foil_position", "sieve_x", "sieve_y"]].copy()
    grid["hole_col"] = np.rint((grid["sieve_x"] - x_origin_cm) / x_spacing_cm).astype(int)
    grid["hole_row"] = np.rint((grid["sieve_y"] - y_origin_cm) / y_spacing_cm).astype(int)

    occupancy = (
        grid.groupby(["foil_position", "hole_row", "hole_col"], as_index=False)
        .size()
        .rename(columns={"size": "hole_population"})
    )
    min_population = int(max(getattr(args, "direct_grid_min_hole_population", 1), 1))
    occupancy = occupancy.loc[occupancy["hole_population"] >= min_population].copy()
    if occupancy.empty:
        raise RuntimeError(
            "No occupied chessboard cells survived direct_grid minimum-population filtering."
        )

    design = occupancy[["foil_position", "hole_row", "hole_col"]].drop_duplicates().copy()
    design["foil_position"] = design["foil_position"].astype(int)
    design["hole_row"] = design["hole_row"].astype(int)
    design["hole_col"] = design["hole_col"].astype(int)
    design["candidate_sieve_x_cm"] = x_origin_cm + design["hole_col"].to_numpy(dtype=np.float64) * x_spacing_cm
    design["candidate_sieve_y_cm"] = y_origin_cm + design["hole_row"].to_numpy(dtype=np.float64) * y_spacing_cm
    design["weak_hole_xptar_center"] = design["candidate_sieve_x_cm"] / float(args.sieve_distance_cm)
    design["weak_hole_yptar_center"] = design["candidate_sieve_y_cm"] / float(args.sieve_distance_cm)
    design["weak_hole_xptar_tol"] = float(tolerance_rad)
    design["weak_hole_yptar_tol"] = float(tolerance_rad)

    per_foil_summary: dict[str, Any] = {}
    for foil_pos, df_foil in occupancy.groupby("foil_position", sort=True):
        foil_pos_int = int(cast(Any, foil_pos))
        per_foil_summary[str(foil_pos_int)] = {
            "row_range": [int(df_foil["hole_row"].min()), int(df_foil["hole_row"].max())],
            "col_range": [int(df_foil["hole_col"].min()), int(df_foil["hole_col"].max())],
            "occupied_holes": int(len(df_foil)),
        }

    meta = {
        "x_spacing_cm": float(x_spacing_cm),
        "y_spacing_cm": float(y_spacing_cm),
        "x_origin_cm": float(x_origin_cm),
        "y_origin_cm": float(y_origin_cm),
        "min_population": int(min_population),
        "per_foil": per_foil_summary,
    }
    return design.sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True), meta


def compute_equal_hole_total_weights(hole_population: pd.Series | np.ndarray) -> np.ndarray:
    populations = np.asarray(hole_population, dtype=np.float64)
    if populations.size == 0:
        return populations.astype(np.float32)
    if np.any(populations <= 0):
        raise ValueError("hole_population must be strictly positive to build inverse-per-hole weights.")
    return (1.0 / populations).astype(np.float32)


def build_mechanical_hole_design_from_grid(
    grid_index: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    x_spacing_mm = getattr(args, "hole_x_spacing_mm", None)
    y_spacing_mm = getattr(args, "hole_y_spacing_mm", None)
    tolerance_mm = getattr(args, "hole_tolerance_mm", None)

    if x_spacing_mm is None or y_spacing_mm is None:
        raise ValueError(
            "Auto-generated hole weak labels require both --hole-x-spacing-mm and --hole-y-spacing-mm."
        )
    if tolerance_mm is None:
        raise ValueError(
            "Auto-generated hole weak labels require --hole-tolerance-mm."
        )
    if grid_index.empty:
        raise RuntimeError("Grid index is empty; cannot auto-generate mechanical hole weak labels.")

    x_spacing_rad = mm_to_target_angle(float(x_spacing_mm), float(args.sieve_distance_cm))
    y_spacing_rad = mm_to_target_angle(float(y_spacing_mm), float(args.sieve_distance_cm))
    tolerance_rad = mm_to_target_angle(float(tolerance_mm), float(args.sieve_distance_cm))

    design = grid_index[["foil_position", "row", "col"]].drop_duplicates().copy()
    design = design.rename(columns={"row": "hole_row", "col": "hole_col"})
    design["foil_position"] = design["foil_position"].astype(int)
    design["hole_row"] = design["hole_row"].astype(int)
    design["hole_col"] = design["hole_col"].astype(int)
    design["weak_hole_xptar_center"] = (
        float(args.hole_origin_xptar) + design["hole_col"].to_numpy(dtype=np.float64) * x_spacing_rad
    )
    design["weak_hole_yptar_center"] = (
        float(args.hole_origin_yptar) + design["hole_row"].to_numpy(dtype=np.float64) * y_spacing_rad
    )
    design["weak_hole_xptar_tol"] = float(tolerance_rad)
    design["weak_hole_yptar_tol"] = float(tolerance_rad)
    return design.sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True)


def build_full_candidate_mechanical_hole_design_from_clusters(
    clustering_results: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_spacing_mm = getattr(args, "hole_x_spacing_mm", None)
    y_spacing_mm = getattr(args, "hole_y_spacing_mm", None)
    tolerance_mm = getattr(args, "hole_tolerance_mm", None)
    if x_spacing_mm is None or y_spacing_mm is None:
        raise ValueError(
            "Auto-generated hole weak labels require both --hole-x-spacing-mm and --hole-y-spacing-mm."
        )
    if tolerance_mm is None:
        raise ValueError("Auto-generated hole weak labels require --hole-tolerance-mm.")

    cluster_centers = extract_cluster_centers(clustering_results)
    if cluster_centers.empty:
        raise RuntimeError("No valid cluster centres found; cannot build mechanical hole candidates.")

    x_spacing_cm = float(x_spacing_mm) / 10.0
    y_spacing_cm = float(y_spacing_mm) / 10.0
    tolerance_rad = mm_to_target_angle(float(tolerance_mm), float(args.sieve_distance_cm))

    x_origin_cm = infer_lattice_origin_cm(
        cluster_centers["cluster_center_x"],
        x_spacing_cm,
        target_angle_to_sieve_cm(float(args.hole_origin_xptar), float(args.sieve_distance_cm)),
    )
    y_origin_cm = infer_lattice_origin_cm(
        cluster_centers["cluster_center_y"],
        y_spacing_cm,
        target_angle_to_sieve_cm(float(args.hole_origin_yptar), float(args.sieve_distance_cm)),
    )

    cluster_centers = cluster_centers.copy()
    cluster_centers["hole_col_est"] = np.rint((cluster_centers["cluster_center_x"] - x_origin_cm) / x_spacing_cm).astype(int)
    cluster_centers["hole_row_est"] = np.rint((cluster_centers["cluster_center_y"] - y_origin_cm) / y_spacing_cm).astype(int)

    design_frames: list[pd.DataFrame] = []
    per_foil_summary: dict[str, Any] = {}
    for foil_pos, df_foil in cluster_centers.groupby("foil_position"):
        foil_pos_int = int(cast(Any, foil_pos))
        min_col = int(df_foil["hole_col_est"].min())
        max_col = int(df_foil["hole_col_est"].max())
        min_row = int(df_foil["hole_row_est"].min())
        max_row = int(df_foil["hole_row_est"].max())

        candidate = pd.MultiIndex.from_product(
            [[foil_pos_int], range(min_row, max_row + 1), range(min_col, max_col + 1)],
            names=["foil_position", "hole_row", "hole_col"],
        ).to_frame(index=False)
        candidate["candidate_sieve_x_cm"] = x_origin_cm + candidate["hole_col"].to_numpy(dtype=np.float64) * x_spacing_cm
        candidate["candidate_sieve_y_cm"] = y_origin_cm + candidate["hole_row"].to_numpy(dtype=np.float64) * y_spacing_cm
        candidate["weak_hole_xptar_center"] = candidate["candidate_sieve_x_cm"] / float(args.sieve_distance_cm)
        candidate["weak_hole_yptar_center"] = candidate["candidate_sieve_y_cm"] / float(args.sieve_distance_cm)
        candidate["weak_hole_xptar_tol"] = float(tolerance_rad)
        candidate["weak_hole_yptar_tol"] = float(tolerance_rad)
        design_frames.append(candidate)
        per_foil_summary[str(foil_pos_int)] = {
            "row_range": [min_row, max_row],
            "col_range": [min_col, max_col],
            "candidate_count": int(len(candidate)),
        }

    design = pd.concat(design_frames, ignore_index=True)
    meta = {
        "x_spacing_cm": float(x_spacing_cm),
        "y_spacing_cm": float(y_spacing_cm),
        "x_origin_cm": float(x_origin_cm),
        "y_origin_cm": float(y_origin_cm),
        "per_foil": per_foil_summary,
    }
    return design.sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True), meta


def build_cluster_to_mechanical_hole_map(
    clustering_results: dict[int, dict[str, Any]],
    hole_design: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cluster_centers = extract_cluster_centers(clustering_results)
    if cluster_centers.empty:
        raise RuntimeError("No cluster centres available for nearest mechanical-hole matching.")

    candidate_design = ensure_candidate_sieve_columns(hole_design, args)

    assignments: list[dict[str, Any]] = []
    per_foil_stats: dict[str, Any] = {}
    for foil_pos, df_foil in cluster_centers.groupby("foil_position"):
        foil_pos_int = int(cast(Any, foil_pos))
        candidates = candidate_design.loc[candidate_design["foil_position"] == foil_pos].copy()
        if candidates.empty:
            raise RuntimeError(f"No mechanical-hole candidates found for foil {foil_pos_int}.")
        candidate_xy = candidates[["candidate_sieve_x_cm", "candidate_sieve_y_cm"]].to_numpy(dtype=np.float64)
        candidate_rowcol = candidates[["hole_row", "hole_col"]].to_numpy(dtype=np.int64)
        center_idx = int(np.argmin(np.sqrt(candidate_xy[:, 0] ** 2 + candidate_xy[:, 1] ** 2)))
        center_ref_x = float(candidate_xy[center_idx, 0])
        center_ref_y = float(candidate_xy[center_idx, 1])
        assignment_mode = str(getattr(args, "cluster_hole_assignment_mode", "nearest"))
        occupancy_penalty_cm = float(max(getattr(args, "cluster_hole_occupancy_penalty_cm", 0.0), 0.0))
        cluster_frame = df_foil.copy()
        cluster_radius = np.sqrt(
            (cluster_frame["cluster_center_x"].to_numpy(dtype=np.float64) - center_ref_x) ** 2
            + (cluster_frame["cluster_center_y"].to_numpy(dtype=np.float64) - center_ref_y) ** 2
        )
        cluster_frame["__center_out_radius"] = cluster_radius
        if assignment_mode == "center_out_penalized":
            cluster_frame = cluster_frame.sort_values(
                ["__center_out_radius", "cluster_center_x", "cluster_center_y", "cluster"],
                ascending=[True, True, True, True],
            ).reset_index(drop=True)

        distances: list[float] = []
        effective_costs: list[float] = []
        occupancies_before: list[int] = []
        reassigned_from_nearest = 0
        hole_usage_counts: dict[tuple[int, int], int] = {}
        for row in cluster_frame.itertuples(index=False):
            cluster_id = int(cast(Any, row.cluster))
            cluster_center_x = float(cast(Any, row.cluster_center_x))
            cluster_center_y = float(cast(Any, row.cluster_center_y))
            delta = candidate_xy - np.array([cluster_center_x, cluster_center_y], dtype=np.float64)
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            nearest_idx = int(np.argmin(dist))
            if assignment_mode == "center_out_penalized":
                occupancy_counts = np.array(
                    [hole_usage_counts.get((int(rc[0]), int(rc[1])), 0) for rc in candidate_rowcol],
                    dtype=np.float64,
                )
                effective_cost = dist + occupancy_penalty_cm * occupancy_counts
                best_idx = int(np.argmin(effective_cost))
                occupancies_before.append(int(occupancy_counts[best_idx]))
                effective_costs.append(float(effective_cost[best_idx]))
            else:
                best_idx = nearest_idx
                occupancies_before.append(int(hole_usage_counts.get((int(candidate_rowcol[best_idx, 0]), int(candidate_rowcol[best_idx, 1])), 0)))
                effective_costs.append(float(dist[best_idx]))
            if best_idx != nearest_idx:
                reassigned_from_nearest += 1
            distances.append(float(dist[best_idx]))
            chosen_key = (int(candidate_rowcol[best_idx, 0]), int(candidate_rowcol[best_idx, 1]))
            hole_usage_counts[chosen_key] = hole_usage_counts.get(chosen_key, 0) + 1
            assignments.append(
                {
                    "foil_position": foil_pos_int,
                    "cluster": cluster_id,
                    "cluster_center_x": cluster_center_x,
                    "cluster_center_y": cluster_center_y,
                    "hole_row": int(candidate_rowcol[best_idx, 0]),
                    "hole_col": int(candidate_rowcol[best_idx, 1]),
                    "matched_sieve_x_cm": float(candidate_xy[best_idx, 0]),
                    "matched_sieve_y_cm": float(candidate_xy[best_idx, 1]),
                    "match_dx_cm": float(cluster_center_x - candidate_xy[best_idx, 0]),
                    "match_dy_cm": float(cluster_center_y - candidate_xy[best_idx, 1]),
                    "match_distance_cm": float(dist[best_idx]),
                    "nearest_match_distance_cm": float(dist[nearest_idx]),
                    "effective_match_cost_cm": float(effective_costs[-1]),
                    "hole_occupancy_before_assignment": int(occupancies_before[-1]),
                    "assignment_mode": assignment_mode,
                }
            )
        per_foil_stats[str(foil_pos_int)] = {
            "n_clusters": int(len(df_foil)),
            "median_match_distance_cm": float(np.median(distances)) if distances else None,
            "max_match_distance_cm": float(np.max(distances)) if distances else None,
            "min_match_distance_cm": float(np.min(distances)) if distances else None,
            "median_effective_match_cost_cm": float(np.median(effective_costs)) if effective_costs else None,
            "reassigned_from_nearest_count": int(reassigned_from_nearest),
            "max_hole_occupancy": int(max(hole_usage_counts.values())) if hole_usage_counts else 0,
        }

    assignment_df = pd.DataFrame(assignments).sort_values(["foil_position", "cluster"]).reset_index(drop=True)
    duplicate_assignments = (
        assignment_df.groupby(["foil_position", "hole_row", "hole_col"]).size().reset_index(name="assigned_clusters")
    )
    summary = {
        "assignment_mode": str(getattr(args, "cluster_hole_assignment_mode", "nearest")),
        "occupancy_penalty_cm": float(max(getattr(args, "cluster_hole_occupancy_penalty_cm", 0.0), 0.0)),
        "per_foil": per_foil_stats,
        "duplicate_mechanical_holes": int((duplicate_assignments["assigned_clusters"] > 1).sum()),
    }
    return assignment_df, summary


def describe_cluster_hole_assignment_source(
    args: argparse.Namespace,
    *,
    generated_from_spacing: bool,
) -> str:
    assignment_mode = str(getattr(args, "cluster_hole_assignment_mode", "nearest"))
    prefix = "generated_from_spacing" if generated_from_spacing else "table"
    if assignment_mode == "center_out_penalized":
        return f"{prefix}_center_out_penalized_cluster_match"
    return f"{prefix}_nearest_cluster_match"


def assign_events_to_mechanical_holes(
    df: pd.DataFrame,
    hole_design: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if df.empty:
        raise RuntimeError("No events available for direct_grid mechanical-hole assignment.")

    candidate_design = ensure_candidate_sieve_columns(hole_design, args)
    assignment_frames: list[pd.DataFrame] = []
    for foil_pos, df_foil in df.groupby("foil_position", sort=True):
        foil_pos_int = int(cast(Any, foil_pos))
        candidates = candidate_design.loc[candidate_design["foil_position"] == foil_pos_int].copy()
        if candidates.empty:
            continue

        event_xy = df_foil[["sieve_x", "sieve_y"]].to_numpy(dtype=np.float64)
        candidate_xy = candidates[["candidate_sieve_x_cm", "candidate_sieve_y_cm"]].to_numpy(dtype=np.float64)
        delta = event_xy[:, None, :] - candidate_xy[None, :, :]
        dist_sq = np.sum(delta * delta, axis=2)
        best_idx = np.argmin(dist_sq, axis=1)

        assigned = df_foil.copy()
        assigned["hole_row"] = candidates.iloc[best_idx]["hole_row"].to_numpy(dtype=np.int64)
        assigned["hole_col"] = candidates.iloc[best_idx]["hole_col"].to_numpy(dtype=np.int64)
        assigned["candidate_sieve_x_cm"] = candidates.iloc[best_idx]["candidate_sieve_x_cm"].to_numpy(dtype=np.float64)
        assigned["candidate_sieve_y_cm"] = candidates.iloc[best_idx]["candidate_sieve_y_cm"].to_numpy(dtype=np.float64)
        assigned["matched_sieve_x_cm"] = assigned["candidate_sieve_x_cm"]
        assigned["matched_sieve_y_cm"] = assigned["candidate_sieve_y_cm"]
        assigned["weak_hole_xptar_center"] = candidates.iloc[best_idx]["weak_hole_xptar_center"].to_numpy(dtype=np.float64)
        assigned["weak_hole_yptar_center"] = candidates.iloc[best_idx]["weak_hole_yptar_center"].to_numpy(dtype=np.float64)
        assigned["weak_hole_xptar_tol"] = candidates.iloc[best_idx]["weak_hole_xptar_tol"].to_numpy(dtype=np.float64)
        assigned["weak_hole_yptar_tol"] = candidates.iloc[best_idx]["weak_hole_yptar_tol"].to_numpy(dtype=np.float64)
        assigned["match_dx_cm"] = assigned["sieve_x"] - assigned["candidate_sieve_x_cm"]
        assigned["match_dy_cm"] = assigned["sieve_y"] - assigned["candidate_sieve_y_cm"]
        assigned["match_distance_cm"] = np.sqrt(np.take_along_axis(dist_sq, best_idx[:, None], axis=1).reshape(-1))
        assignment_frames.append(assigned)

    if not assignment_frames:
        raise RuntimeError("No direct_grid event assignments were produced.")

    assigned_raw = pd.concat(assignment_frames, ignore_index=True)
    assigned = assigned_raw.copy()
    x_spacing_mm = getattr(args, "hole_x_spacing_mm", None)
    y_spacing_mm = getattr(args, "hole_y_spacing_mm", None)
    max_match_distance_cm = getattr(args, "direct_grid_max_match_distance_cm", None)
    if max_match_distance_cm is None and x_spacing_mm is not None and y_spacing_mm is not None:
        max_match_distance_cm = 0.5 * math.hypot(float(x_spacing_mm) / 10.0, float(y_spacing_mm) / 10.0)
    if max_match_distance_cm is not None:
        assigned = assigned.loc[assigned["match_distance_cm"] <= float(max_match_distance_cm)].copy()
    if assigned.empty:
        raise RuntimeError("All direct_grid assignments were rejected by match-distance filtering.")

    n_after_distance = int(len(assigned))

    occupancy = (
        assigned.groupby(["foil_position", "hole_row", "hole_col"], as_index=False)
        .size()
        .rename(columns={"size": "hole_population"})
    )
    min_population = int(max(getattr(args, "direct_grid_min_hole_population", 1), 1))
    valid_holes = occupancy.loc[occupancy["hole_population"] >= min_population].copy()
    if valid_holes.empty:
        raise RuntimeError(
            "All direct_grid holes were removed by minimum-population filtering; no training labels remain."
        )

    max_holes_per_foil = getattr(args, "direct_grid_max_holes_per_foil", None)
    per_foil_cap_summary: dict[str, Any] = {}
    if max_holes_per_foil is not None:
        max_holes_per_foil = int(max(max_holes_per_foil, 1))
        hole_quality = (
            assigned.groupby(["foil_position", "hole_row", "hole_col"], as_index=False)
            .agg(median_match_distance_cm=("match_distance_cm", "median"))
        )
        ranked_holes = valid_holes.merge(
            hole_quality,
            on=["foil_position", "hole_row", "hole_col"],
            how="left",
        )
        kept_holes: list[pd.DataFrame] = []
        for foil_pos, df_foil in ranked_holes.groupby("foil_position", sort=True):
            foil_pos_int = int(cast(Any, foil_pos))
            ranked = df_foil.sort_values(
                ["hole_population", "median_match_distance_cm", "hole_row", "hole_col"],
                ascending=[False, True, True, True],
            ).reset_index(drop=True)
            kept = ranked.head(max_holes_per_foil).copy()
            kept_holes.append(kept)
            per_foil_cap_summary[str(foil_pos_int)] = {
                "holes_before_cap": int(len(ranked)),
                "holes_after_cap": int(len(kept)),
                "min_population_kept": int(kept["hole_population"].min()) if not kept.empty else None,
                "max_population_kept": int(kept["hole_population"].max()) if not kept.empty else None,
            }
        valid_holes = pd.concat(kept_holes, ignore_index=True)

    assigned = assigned.merge(
        valid_holes,
        on=["foil_position", "hole_row", "hole_col"],
        how="inner",
    )
    if assigned.empty:
        raise RuntimeError("No events survived direct_grid minimum-population filtering.")

    filtered_design = candidate_design.merge(
        valid_holes[["foil_position", "hole_row", "hole_col"]],
        on=["foil_position", "hole_row", "hole_col"],
        how="inner",
    ).drop_duplicates(subset=["foil_position", "hole_row", "hole_col"])

    cluster_lookup = valid_holes[["foil_position", "hole_row", "hole_col", "hole_population"]].sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True).copy()
    cluster_lookup["cluster"] = cluster_lookup.groupby("foil_position").cumcount().astype(int)

    cluster_centers = (
        assigned.groupby(["foil_position", "hole_row", "hole_col"], as_index=False)
        .agg(
            cluster_center_x=("sieve_x", "median"),
            cluster_center_y=("sieve_y", "median"),
        )
    )
    cluster_lookup = cluster_lookup.merge(
        cluster_centers,
        on=["foil_position", "hole_row", "hole_col"],
        how="left",
    )
    assigned = assigned.merge(
        cluster_lookup,
        on=["foil_position", "hole_row", "hole_col", "hole_population"],
        how="left",
    )
    assigned["is_noise"] = False

    per_foil_stats: dict[str, Any] = {}
    for foil_pos, df_foil in assigned.groupby("foil_position", sort=True):
        foil_pos_int = int(cast(Any, foil_pos))
        distances = df_foil["match_distance_cm"].to_numpy(dtype=np.float64)
        per_foil_stats[str(foil_pos_int)] = {
            "n_events": int(len(df_foil)),
            "n_holes": int(df_foil["cluster"].nunique()),
            "median_match_distance_cm": float(np.median(distances)) if len(distances) else None,
            "max_match_distance_cm": float(np.max(distances)) if len(distances) else None,
            "min_match_distance_cm": float(np.min(distances)) if len(distances) else None,
        }

    summary = {
        "per_foil": per_foil_stats,
        "min_population": int(min_population),
        "max_match_distance_cm": float(max_match_distance_cm) if max_match_distance_cm is not None else None,
        "max_holes_per_foil": int(max_holes_per_foil) if max_holes_per_foil is not None else None,
        "per_foil_hole_cap": per_foil_cap_summary,
        "dropped_events_far_from_hole": int(len(assigned_raw) - n_after_distance),
        "dropped_events_below_population": int(n_after_distance - len(assigned)),
    }
    return assigned.reset_index(drop=True), filtered_design.reset_index(drop=True), summary


def resolve_mechanical_hole_design(
    args: argparse.Namespace,
    clustering_results: dict[int, dict[str, Any]],
    output_csv: Path,
) -> tuple[pd.DataFrame, str, str | None, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if args.hole_design_table:
        design = load_mechanical_hole_design(args)
        design_meta = {
            "x_spacing_cm": None,
            "y_spacing_cm": None,
            "x_origin_cm": None,
            "y_origin_cm": None,
            "per_foil": {},
        }
        cluster_hole_map, match_summary = build_cluster_to_mechanical_hole_map(clustering_results, design, args)
        return design, describe_cluster_hole_assignment_source(args, generated_from_spacing=False), str(args.hole_design_table), cluster_hole_map, design_meta, match_summary

    design, design_meta = build_full_candidate_mechanical_hole_design_from_clusters(clustering_results, args)
    resolved_path = Path(args.resolved_hole_design_csv) if args.resolved_hole_design_csv else output_csv.with_name(
        output_csv.stem + "_resolved_hole_design.csv"
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(resolved_path, index=False)
    cluster_hole_map, match_summary = build_cluster_to_mechanical_hole_map(clustering_results, design, args)
    return design, describe_cluster_hole_assignment_source(args, generated_from_spacing=True), str(resolved_path), cluster_hole_map, design_meta, match_summary


def resolve_direct_grid_hole_design(
    args: argparse.Namespace,
    df: pd.DataFrame,
    output_csv: Path,
) -> tuple[pd.DataFrame, str, str | None, dict[str, Any]]:
    if args.hole_design_table:
        design = ensure_candidate_sieve_columns(load_mechanical_hole_design(args), args)
        design_meta = {
            "x_spacing_cm": None,
            "y_spacing_cm": None,
            "x_origin_cm": None,
            "y_origin_cm": None,
            "min_population": int(max(getattr(args, "direct_grid_min_hole_population", 1), 1)),
            "per_foil": {},
        }
        return design, "table_direct_grid_assignment", str(args.hole_design_table), design_meta

    design, design_meta = build_observed_mechanical_hole_design_from_event_grid(df, args)
    resolved_path = Path(args.resolved_hole_design_csv) if args.resolved_hole_design_csv else output_csv.with_name(
        output_csv.stem + "_resolved_hole_design.csv"
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(resolved_path, index=False)
    return design, "generated_from_spacing_direct_grid", str(resolved_path), design_meta


def load_foil_center_map(args: argparse.Namespace, foil_positions: list[int]) -> dict[int, float]:
    if args.foil_centers_json:
        with open(args.foil_centers_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        center_map = {int(k): float(v) for k, v in payload.items()}
    else:
        center_map = dict(DEFAULT_25521_FOIL_CENTERS)

    missing = [foil for foil in foil_positions if foil not in center_map]
    if missing:
        raise ValueError(
            "Foil center map is missing foil positions: " + ", ".join(str(v) for v in missing)
        )
    return center_map


def load_and_standardise_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    root_file = cast(Any, uproot.open(args.root_file))
    tree = cast(Any, root_file[args.tree_name])
    branch_map = resolve_root_branch_map(tree)
    df = cast(pd.DataFrame, tree.arrays(list(branch_map.keys()), library="pd"))

    if args.max_events is not None:
        df = df.iloc[: args.max_events].copy()

    df = df.rename(columns=branch_map)
    if "run_id" not in df.columns:
        df["run_id"] = infer_run_id(args.root_file)

    required_cols = REQUIRED_CANONICAL_BRANCHES + ["run_id"]
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
    cluster_hole_map: pd.DataFrame,
    hole_design: pd.DataFrame,
    hole_design_source: str,
    resolved_hole_design_path: str | None,
    design_meta: dict[str, Any],
    match_summary: dict[str, Any],
    foil_center_map: dict[int, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labeled_frames: list[pd.DataFrame] = []
    if cluster_hole_map.empty:
        raise RuntimeError("Nearest mechanical-hole matching returned no assignments; cannot build Stage-2 labels.")

    for foil_pos, result in sorted(clustering_results.items()):
        df_foil = result["df"].copy()
        df_foil = df_foil.loc[df_foil["cluster"] != -1].copy()
        if df_foil.empty:
            continue

        centers = cluster_hole_map.loc[
            cluster_hole_map["foil_position"] == foil_pos,
            [
                "foil_position",
                "cluster",
                "hole_row",
                "hole_col",
                "matched_sieve_x_cm",
                "matched_sieve_y_cm",
                "match_dx_cm",
                "match_dy_cm",
                "match_distance_cm",
            ],
        ]
        df_foil = df_foil.merge(centers, on=["foil_position", "cluster"], how="inner")
        if df_foil.empty:
            continue

        cluster_stats = (
            df_foil.groupby(["foil_position", "cluster", "hole_row", "hole_col"], as_index=False)
            .agg(hole_population=("cluster", "size"))
        )
        cluster_stats = cluster_stats.merge(
            hole_design,
            on=["foil_position", "hole_row", "hole_col"],
            how="left",
        )
        missing_design = cluster_stats[
            cluster_stats[["weak_hole_xptar_center", "weak_hole_yptar_center"]].isna().any(axis=1)
        ]
        if not missing_design.empty:
            missing_examples = missing_design[["foil_position", "hole_row", "hole_col"]].head(10).to_dict(orient="records")
            raise RuntimeError(
                "Mechanical hole design table does not cover all clustered holes. Missing examples: "
                f"{missing_examples}"
            )

        df_foil = df_foil.merge(
            cluster_stats,
            on=["foil_position", "cluster", "hole_row", "hole_col"],
            how="left",
        )
        df_foil["foil_ytar_center"] = float(foil_center_map[int(foil_pos)])
        df_foil["foil_ytar_tol"] = 0.0
        # Deprecated compatibility columns for older analysis helpers.
        df_foil["weak_foil_ytar_center"] = df_foil["foil_ytar_center"]
        df_foil["weak_foil_ytar_tol"] = df_foil["foil_ytar_tol"]
        df_foil["foil_population"] = len(df_foil)
        labeled_frames.append(df_foil)

    if not labeled_frames:
        raise RuntimeError("No clustered foil events survived the event-level weak-label merge.")

    labeled = pd.concat(labeled_frames, ignore_index=True)
    labeled["hole_id"] = labeled["foil_position"].astype(int) * 1000 + labeled["cluster"].astype(int)

    labeled["weak_label_weight"] = compute_equal_hole_total_weights(labeled["hole_population"])

    labeled["hole_row"] = labeled["hole_row"].astype(int)
    labeled["hole_col"] = labeled["hole_col"].astype(int)
    labeled["foil_position"] = labeled["foil_position"].astype(int)
    labeled["run_id"] = labeled["run_id"].astype(int)

    summary = {
        "n_labeled_events": int(len(labeled)),
        "hole_label_source": hole_design_source,
        "hole_design_table": str(args.hole_design_table) if args.hole_design_table else None,
        "resolved_hole_design_csv": resolved_hole_design_path,
        "ytar_label_source": "strong_fixed_foil_centers",
        "foil_centers_cm": {str(int(k)): float(v) for k, v in sorted(foil_center_map.items())},
        "foil_counts": {str(k): int(v) for k, v in labeled["foil_position"].value_counts().sort_index().items()},
        "hole_count_total": int(labeled["hole_id"].nunique()),
        "hole_spacing_mm": {
            "x": float(args.hole_x_spacing_mm) if args.hole_x_spacing_mm is not None else None,
            "y": float(args.hole_y_spacing_mm) if args.hole_y_spacing_mm is not None else None,
        },
        "hole_tolerance_mm": float(args.hole_tolerance_mm) if args.hole_tolerance_mm is not None else None,
        "sieve_distance_cm": float(args.sieve_distance_cm),
        "cluster_count_per_foil": {
            str(int(foil)): int(result["n_clusters"])
            for foil, result in sorted(clustering_results.items())
        },
        "cluster_count_before_post_per_foil": {
            str(int(foil)): int(result["n_clusters_before_post"])
            for foil, result in sorted(clustering_results.items())
        },
        "mechanical_candidate_grid": design_meta,
        "cluster_to_mechanical_match": match_summary,
        "weighting_scheme": "equal_hole_total_weight",
    }
    return labeled, summary


def build_event_level_labels_from_direct_assignments(
    assigned_events: pd.DataFrame,
    hole_design: pd.DataFrame,
    hole_design_source: str,
    resolved_hole_design_path: str | None,
    design_meta: dict[str, Any],
    match_summary: dict[str, Any],
    foil_center_map: dict[int, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if assigned_events.empty:
        raise RuntimeError("No direct_grid assignments survived; cannot build Stage-2 labels.")

    labeled = assigned_events.copy()
    required_cols = [
        "weak_hole_xptar_center",
        "weak_hole_yptar_center",
        "weak_hole_xptar_tol",
        "weak_hole_yptar_tol",
        "hole_population",
        "cluster",
        "cluster_center_x",
        "cluster_center_y",
    ]
    missing_cols = [col for col in required_cols if col not in labeled.columns]
    if missing_cols:
        raise RuntimeError(
            "Direct-grid assignments are missing required columns: " + ", ".join(missing_cols)
        )

    labeled["foil_ytar_center"] = labeled["foil_position"].map(foil_center_map).astype(float)
    labeled["foil_ytar_tol"] = 0.0
    labeled["weak_foil_ytar_center"] = labeled["foil_ytar_center"]
    labeled["weak_foil_ytar_tol"] = labeled["foil_ytar_tol"]
    foil_population = labeled.groupby("foil_position").size().rename("foil_population").reset_index()
    labeled = labeled.drop(columns=["foil_population"], errors="ignore").merge(foil_population, on="foil_position", how="left")
    labeled["hole_id"] = labeled["foil_position"].astype(int) * 1000 + labeled["cluster"].astype(int)

    labeled["weak_label_weight"] = compute_equal_hole_total_weights(labeled["hole_population"])

    labeled["hole_row"] = labeled["hole_row"].astype(int)
    labeled["hole_col"] = labeled["hole_col"].astype(int)
    labeled["foil_position"] = labeled["foil_position"].astype(int)
    labeled["run_id"] = labeled["run_id"].astype(int)

    hole_design_used = hole_design.drop_duplicates(subset=["foil_position", "hole_row", "hole_col"])
    summary = {
        "n_labeled_events": int(len(labeled)),
        "hole_label_source": hole_design_source,
        "hole_design_table": str(args.hole_design_table) if args.hole_design_table else None,
        "resolved_hole_design_csv": resolved_hole_design_path,
        "ytar_label_source": "strong_fixed_foil_centers",
        "foil_centers_cm": {str(int(k)): float(v) for k, v in sorted(foil_center_map.items())},
        "foil_counts": {str(k): int(v) for k, v in labeled["foil_position"].value_counts().sort_index().items()},
        "hole_count_total": int(labeled["hole_id"].nunique()),
        "hole_spacing_mm": {
            "x": float(args.hole_x_spacing_mm) if args.hole_x_spacing_mm is not None else None,
            "y": float(args.hole_y_spacing_mm) if args.hole_y_spacing_mm is not None else None,
        },
        "hole_tolerance_mm": float(args.hole_tolerance_mm) if args.hole_tolerance_mm is not None else None,
        "sieve_distance_cm": float(args.sieve_distance_cm),
        "cluster_count_per_foil": {
            str(int(cast(Any, foil))): int(df_foil["cluster"].nunique())
            for foil, df_foil in labeled.groupby("foil_position", sort=True)
        },
        "cluster_count_before_post_per_foil": {
            str(int(cast(Any, foil))): int(df_foil["cluster"].nunique())
            for foil, df_foil in labeled.groupby("foil_position", sort=True)
        },
        "mechanical_candidate_grid": design_meta,
        "cluster_to_mechanical_match": match_summary,
        "resolved_hole_count": int(len(hole_design_used)),
        "weighting_scheme": "equal_hole_total_weight",
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
    foil_positions = sorted(int(v) for v in df["foil_position"].dropna().unique() if v != -1)
    foil_center_map = load_foil_center_map(args, foil_positions)
    print("Foil counts after classification:")
    print(df["foil_position"].value_counts().sort_index().to_string())

    if args.clustering_method == "direct_grid":
        hole_design, hole_design_source, resolved_hole_design_path, design_meta = resolve_direct_grid_hole_design(
            args,
            df,
            output_csv,
        )
        assigned_events, hole_design, match_summary = assign_events_to_mechanical_holes(df, hole_design, args)
        print(f"Hole weak-label source       : {hole_design_source}")
        if resolved_hole_design_path:
            print(f"Resolved hole design CSV     : {resolved_hole_design_path}")
        for foil_pos, df_foil in assigned_events.groupby("foil_position", sort=True):
            print(
                f"Foil {int(cast(Any, foil_pos))}: direct-grid holes={df_foil['cluster'].nunique()}, "
                f"events={len(df_foil):,}"
            )
        labeled, summary = build_event_level_labels_from_direct_assignments(
            assigned_events,
            hole_design,
            hole_design_source,
            resolved_hole_design_path,
            design_meta,
            match_summary,
            foil_center_map,
            args,
        )
    else:
        clustering_results = cluster_each_foil(df, args)
        for foil_pos, result in sorted(clustering_results.items()):
            print(
                f"Foil {foil_pos}: clusters {result['n_clusters_before_post']} -> {result['n_clusters']} "
                f"(post), events={len(result['df']):,}"
            )

        hole_design, hole_design_source, resolved_hole_design_path, cluster_hole_map, design_meta, match_summary = resolve_mechanical_hole_design(args, clustering_results, output_csv)
        print(f"Hole weak-label source       : {hole_design_source}")
        if resolved_hole_design_path:
            print(f"Resolved hole design CSV     : {resolved_hole_design_path}")

        labeled, summary = build_event_level_labels(
            clustering_results,
            cluster_hole_map,
            hole_design,
            hole_design_source,
            resolved_hole_design_path,
            design_meta,
            match_summary,
            foil_center_map,
            args,
        )
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
