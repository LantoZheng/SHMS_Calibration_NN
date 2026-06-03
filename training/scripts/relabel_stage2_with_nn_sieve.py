#!/usr/bin/env python3
"""
Rebuild Stage-2 labels using NN-predicted sieve positions instead of HCANA.

This breaks the circular dependency where labels derived from HCANA's
project_to_sieve are used to train a NN, then the NN is compared against HCANA.

Workflow:
  1. Load best Stage-2 checkpoint
  2. Run inference on the full labelled dataset
  3. Replace HCANA sieve_x/sieve_y with NN-predicted sieve positions
  4. Re-run HDBSCAN clustering on NN sieve positions
  5. Match clusters to mechanical hole design grid
  6. Export new labelled CSV

The only "ground truth" used from HCANA is:
  - P_gtr_y for foil classification (independent of transport)
  - Foil centers (hard physical constraint, not reconstruction)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.models import build_model_from_config
from training.scripts.build_stage2_labels_from_25521_fullroot import (
    HDBSCANConfig,
    build_event_level_labels,
    build_full_candidate_mechanical_hole_design_from_clusters,
    cluster_each_foil,
    compute_equal_hole_total_weights,
    describe_cluster_hole_assignment_source,
    resolve_mechanical_hole_design,
)

_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOIL_CENTERS = {0: 10.0, 1: 0.0, 2: -10.0}
_FOIL_YTAR_TOLERANCES = {0: 0.4, 1: 0.8, 2: 0.5}  # based on achievable NN sigma68 per foil
_SIEVE_DISTANCE_CM = 253.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild labels with NN-predicted sieve positions")
    parser.add_argument("--checkpoint", required=True, help="Path to best_finetune.pth")
    parser.add_argument("--input-csv", required=True, help="Path to existing labelled Stage-2 CSV")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for NN-relabelled data")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--resolved-hole-design-csv", default=None, help="Optional resolved hole design output")
    parser.add_argument("--device", default="cuda", help="cpu / cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hole-x-spacing-mm", type=float, default=25.0)
    parser.add_argument("--hole-y-spacing-mm", type=float, default=16.4)
    parser.add_argument("--hole-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--sieve-distance-cm", type=float, default=253.0)
    parser.add_argument("--clustering-method", choices=["hdbscan", "two_entry"], default="hdbscan")
    parser.add_argument("--cluster-hole-assignment-mode", choices=["nearest", "center_out_penalized"],
                        default="center_out_penalized")
    parser.add_argument("--cluster-hole-occupancy-penalty-cm", type=float, default=0.35)
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
    parser.add_argument("--scaler-bundle", default=None, help="Optional override for scaler bundle path")
    parser.add_argument("--ytar-foil-check-margin-cm", type=float, default=3.0,
                        help="Margin in cm for NN ytar-based foil reassignment (default 3.0, 0=disable)")
    return parser.parse_args()


@torch.no_grad()
def predict_physical(
    model: torch.nn.Module,
    dataset: Stage2RootDataset,
    scaler: ScalerBundle,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    blocks = []
    for batch in loader:
        inp = batch["inputs"].to(device)
        out = model(inp)
        blocks.append(torch.cat([out[k] for k in _TARGET_KEYS], dim=1).detach().cpu().numpy())
    pred_scaled = np.concatenate(blocks, axis=0).astype(np.float64)
    return scaler.inverse_transform_Y(pred_scaled)


def classify_foils_from_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Use existing foil_position column (from HCANA P_gtr_y classification)."""
    df = df.loc[df["foil_position"].notna()].copy()
    df["foil_position"] = df["foil_position"].astype(int)
    df = df.loc[df["foil_position"].isin([0, 1, 2])].copy()
    return df.reset_index(drop=True)


def correct_foil_assignment_by_nn_ytar(
    df: pd.DataFrame,
    margin_cm: float = 3.0,
) -> pd.DataFrame:
    """
    Reassign events to a different foil if NN-predicted ytar strongly disagrees
    with the current foil_position (which originally came from HCANA's P_gtr_y).

    For each event:
      - Compute distance from nn_ytar to each foil center (+10, 0, -10 cm)
      - If nn_ytar is closest to a different foil AND the gap exceeds margin_cm,
        reassign the event to that foil.

    This uses the hard physical constraint that foils are at known z positions,
    independent of any reconstruction.
    """
    if "nn_ytar" not in df.columns:
        return df

    centers = np.array([_FOIL_CENTERS[f] for f in [0, 1, 2]], dtype=np.float64)
    foil_ids = np.array([0, 1, 2], dtype=np.int32)

    nn_ytar = df["nn_ytar"].to_numpy(dtype=np.float64)
    current_foil = df["foil_position"].to_numpy(dtype=np.int32)

    # Distance to each foil center: shape (N, 3)
    dists = np.abs(nn_ytar[:, None] - centers[None, :])
    best_idx = np.argmin(dists, axis=1)
    best_foil = foil_ids[best_idx]
    best_dist = dists[np.arange(len(dists)), best_idx]

    # Distance to current foil center
    current_center = np.array([_FOIL_CENTERS.get(int(f), 0.0) for f in current_foil])
    dist_to_current = np.abs(nn_ytar - current_center)

    # Reassign if: best foil differs AND current distance exceeds margin
    reassign_mask = (best_foil != current_foil) & (dist_to_current > margin_cm)

    n_reassign = int(reassign_mask.sum())
    if n_reassign > 0:
        # Count per foil
        from_counts = {int(f): int((current_foil[reassign_mask] == f).sum()) for f in [0, 1, 2] if (current_foil[reassign_mask] == f).sum() > 0}
        to_counts = {int(f): int((best_foil[reassign_mask] == f).sum()) for f in [0, 1, 2] if (best_foil[reassign_mask] == f).sum() > 0}
        print(f"  [ytar foil check] Reassigned {n_reassign} events (margin={margin_cm} cm)")
        print(f"    From foils: {from_counts}")
        print(f"    To foils:   {to_counts}")
        df = df.copy()
        df.loc[reassign_mask, "foil_position"] = best_foil[reassign_mask]
        df["foil_position"] = df["foil_position"].astype(int)

    return df


def add_nn_sieve_columns(df: pd.DataFrame, pred_phys: np.ndarray, sieve_distance_cm: float) -> pd.DataFrame:
    df = df.copy()
    df["nn_xptar"] = pred_phys[:, 1]
    df["nn_yptar"] = pred_phys[:, 2]
    df["nn_ytar"] = pred_phys[:, 3]
    df["nn_sieve_x"] = df["nn_xptar"].to_numpy(dtype=np.float64) * sieve_distance_cm
    df["nn_sieve_y"] = df["nn_yptar"].to_numpy(dtype=np.float64) * sieve_distance_cm
    return df


def prepare_clustering_input_from_nn(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sieve_x/sieve_y with NN predictions for clustering, keep all original columns."""
    df_cluster = df.copy()
    df_cluster["sieve_x"] = df_cluster["nn_sieve_x"]
    df_cluster["sieve_y"] = df_cluster["nn_sieve_y"]
    return df_cluster


def build_nn_relabelled_event_labels(
    clustering_results: dict[int, dict[str, Any]],
    cluster_hole_map: pd.DataFrame,
    hole_design: pd.DataFrame,
    hole_design_source: str,
    resolved_hole_design_path: str | None,
    design_meta: dict[str, Any],
    match_summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build event-level labels from NN-based clustering + mechanical hole matching."""
    labeled_frames = []
    for foil_pos in sorted(clustering_results.keys()):
        df_foil = clustering_results[foil_pos]["df"].copy()
        df_foil = df_foil.loc[df_foil["cluster"] != -1].copy()
        if df_foil.empty:
            continue

        # Drop old HCANA-based hole assignments — we want NN-based reassignment
        for drop_col in ["hole_row", "hole_col", "hole_id", "hole_population",
                          "matched_sieve_x_cm", "matched_sieve_y_cm",
                          "match_dx_cm", "match_dy_cm", "match_distance_cm",
                          "candidate_sieve_x_cm", "candidate_sieve_y_cm",
                          "weak_hole_xptar_center", "weak_hole_yptar_center",
                          "weak_hole_xptar_tol", "weak_hole_yptar_tol",
                          "foil_ytar_center", "foil_ytar_tol",
                          "weak_foil_ytar_center", "weak_foil_ytar_tol",
                          "foil_population", "weak_label_weight",
                          "cluster_center_x", "cluster_center_y"]:
            if drop_col in df_foil.columns:
                df_foil = df_foil.drop(columns=[drop_col])

        foil_design = hole_design.loc[hole_design["foil_position"] == foil_pos].copy()
        foil_map = cluster_hole_map.loc[cluster_hole_map["foil_position"] == foil_pos].copy()
        if foil_map.empty:
            print(f"  [WARN] foil {foil_pos}: no cluster→hole map entries, skipping")
            continue

        # Merge cluster→hole assignment
        map_cols = ["cluster", "hole_row", "hole_col", "matched_sieve_x_cm", "matched_sieve_y_cm",
                     "match_dx_cm", "match_dy_cm", "match_distance_cm",
                     "cluster_center_x", "cluster_center_y"]
        events = df_foil.merge(
            foil_map[[c for c in map_cols if c in foil_map.columns]],
            on="cluster", how="inner",
        )

        # Merge hole design (weak labels for xptar/yptar)
        design_merge_cols = ["foil_position", "hole_row", "hole_col",
                              "weak_hole_xptar_center", "weak_hole_yptar_center",
                              "weak_hole_xptar_tol", "weak_hole_yptar_tol"]
        if "candidate_sieve_x_cm" in foil_design.columns:
            design_merge_cols += ["candidate_sieve_x_cm", "candidate_sieve_y_cm"]
        events = events.merge(
            foil_design[design_merge_cols],
            on=["foil_position", "hole_row", "hole_col"], how="left",
        )

        # Compute hole population (per NN cluster)
        pop = events.groupby(["foil_position", "hole_row", "hole_col"]).size().reset_index(name="hole_population")
        events = events.merge(pop, on=["foil_position", "hole_row", "hole_col"], how="left")

        # Foil ytar labels (hard physics with per-foil tolerance)
        events["foil_ytar_center"] = float(_FOIL_CENTERS[int(foil_pos)])
        events["foil_ytar_tol"] = float(_FOIL_YTAR_TOLERANCES.get(int(foil_pos), 0.0))
        events["foil_population"] = len(events)

        labeled_frames.append(events)

    if not labeled_frames:
        raise RuntimeError("No events survived NN re-labelling pipeline.")

    labeled = pd.concat(labeled_frames, ignore_index=True)
    labeled["hole_id"] = labeled["foil_position"].astype(int) * 1000 + labeled["cluster"].astype(int)

    labeled["weak_label_weight"] = compute_equal_hole_total_weights(labeled["hole_population"])

    for col in ["hole_row", "hole_col", "foil_position", "run_id"]:
        if col in labeled.columns:
            labeled[col] = labeled[col].astype(int)

    summary = {
        "n_labeled_events": int(len(labeled)),
        "hole_label_source": f"nn_predicted_sieve_{hole_design_source}",
        "resolved_hole_design_csv": resolved_hole_design_path,
        "ytar_label_source": "strong_fixed_foil_centers",
        "foil_centers_cm": {str(k): float(v) for k, v in _FOIL_CENTERS.items()},
        "foil_counts": {str(k): int(v) for k, v in labeled["foil_position"].value_counts().sort_index().items()},
        "hole_count_total": int(labeled["hole_id"].nunique()),
        "hole_spacing_mm": {"x": 25.0, "y": 16.4},
        "hole_tolerance_mm": 3.0,
        "sieve_distance_cm": _SIEVE_DISTANCE_CM,
        "cluster_count_per_foil": {
            str(foil): int(result["n_clusters"])
            for foil, result in sorted(clustering_results.items())
        },
        "mechanical_candidate_grid": design_meta,
        "cluster_to_mechanical_match": match_summary,
        "weighting_scheme": "equal_hole_total_weight",
    }
    return labeled, summary


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    input_csv_path = Path(args.input_csv).resolve()
    output_csv_path = Path(args.output_csv).resolve()
    summary_json = Path(args.summary_json) if args.summary_json else output_csv_path.with_name(
        output_csv_path.stem + "_summary.json"
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = dict(cfg.get("model", {}))
    model = build_model_from_config(model_cfg, input_dim=int(model_cfg.get("input_dim", 5)))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    scaler_rel = args.scaler_bundle or cfg.get("pretrained", {}).get("scaler_bundle_path")
    scaler_path = Path(scaler_rel) if os.path.isabs(str(scaler_rel)) else Path(checkpoint_path).parent.parent.parent / scaler_rel
    if not scaler_path.exists():
        # Fallback: try relative to repo root
        scaler_path = _REPO_ROOT / scaler_rel
    scaler = ScalerBundle.load(str(scaler_path))

    # ---- Load data ----
    dcfg = cfg.get("data", {})
    dataset = Stage2RootDataset(
        data_source=str(input_csv_path),
        tree_name=dcfg.get("tree_name", "T"),
        scaler_bundle=scaler,
        feature_schema=dcfg.get("feature_schema", ["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"]),
        branch_map=dcfg.get("branch_map", {}),
        label_map=dcfg.get("label_map", {}),
        metadata_cols=dcfg.get("metadata_cols", {}),
        weight_col=dcfg.get("weight_col", None),
        fry_mode=dcfg.get("fry_mode", "direct_or_proxy"),
        direct_fry_branch=dcfg.get("direct_fry_branch", None),
        fry_proxy_branches=dcfg.get("fry_proxy_branches", []),
        cuts=dcfg.get("cuts", {}),
        max_events=args.max_events,
    )
    print(f"Loaded {len(dataset)} events")

    # ---- NN inference ----
    print("Running NN inference on full dataset...")
    pred_phys = predict_physical(model, dataset, scaler, device, args.batch_size)

    # ---- Build NN sieve dataframe ----
    df_orig = dataset.df.reset_index(drop=True).copy()
    df = classify_foils_from_existing(df_orig)
    df = add_nn_sieve_columns(df, pred_phys, float(args.sieve_distance_cm))

    # ---- ytar-based foil consistency check ----
    ytar_margin = float(getattr(args, "ytar_foil_check_margin_cm", 0.0))
    if ytar_margin > 0:
        df = correct_foil_assignment_by_nn_ytar(df, margin_cm=ytar_margin)

    print(f"Events after foil classification: {len(df)} ({df['foil_position'].value_counts().sort_index().to_dict()})")

    # ---- Re-cluster using NN sieve positions ----
    df_cluster = prepare_clustering_input_from_nn(df)
    print(f"Re-clustering with NN-predicted sieve positions (method={args.clustering_method})...")

    clustering_args = SimpleNamespace(
        clustering_method=args.clustering_method,
        cluster_merge_distance=0.85,
        cluster_max_size_ratio=3.5,
        sieve_distance_cm=float(args.sieve_distance_cm),
        hole_x_spacing_mm=float(args.hole_x_spacing_mm),
        hole_y_spacing_mm=float(args.hole_y_spacing_mm),
        hole_tolerance_mm=float(args.hole_tolerance_mm),
        hole_origin_xptar=0.0,
        hole_origin_yptar=0.0,
        hole_design_table=None,
        resolved_hole_design_csv=str(args.resolved_hole_design_csv) if args.resolved_hole_design_csv else None,
        cluster_hole_assignment_mode=args.cluster_hole_assignment_mode,
        cluster_hole_occupancy_penalty_cm=float(args.cluster_hole_occupancy_penalty_cm),
    )

    clustering_results = cluster_each_foil(df_cluster, clustering_args)
    for foil, result in sorted(clustering_results.items()):
        print(f"  Foil {foil}: clusters {result['n_clusters_before_post']} -> {result['n_clusters']} (post), events={len(result['df']):,}")

    # ---- Match to mechanical holes ----
    output_csv = Path(str(args.output_csv))
    hole_design, hole_design_source, resolved_path, cluster_hole_map, design_meta, match_summary = resolve_mechanical_hole_design(
        clustering_args, clustering_results, output_csv
    )
    print(f"Hole label source: {hole_design_source}")
    print(f"Duplicate mechanical holes: {match_summary.get('duplicate_mechanical_holes', '?')}")

    # ---- Build final labels ----
    foil_center_map = dict(_FOIL_CENTERS)
    labeled, summary = build_nn_relabelled_event_labels(
        clustering_results,
        cluster_hole_map,
        hole_design, hole_design_source,
        resolved_path, design_meta, match_summary,
    )

    # ---- Export ----
    labeled = labeled.sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True)
    labeled.to_csv(output_csv_path, index=False)
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nNN-relabelled Stage-2 CSV saved: {output_csv_path}")
    print(f"Summary JSON saved: {summary_json}")
    print(f"Labeled events: {len(labeled):,}")
    print(f"Unique hole_id: {labeled['hole_id'].nunique():,}")
    print(f"Foil counts: {labeled['foil_position'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
