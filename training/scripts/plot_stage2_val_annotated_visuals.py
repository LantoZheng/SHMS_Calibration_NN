#!/usr/bin/env python3
"""Comprehensive labeling + reconstruction visuals with validation-hole highlighting.

Outputs (all saved to --output-dir):
  1. labeling_results_3x2_val_annotated.png
     Left: clustered events (train=color, val=black edge / red edge)
     Right: cluster→label matches with validation holes highlighted
  2. sieve_pattern_3x2_val_annotated.png
     Six-panel sieve pattern: NN + ROOT per foil, validation events overlaid
  3. ytar_val_annotated.png
     ztar distributions with holdout-hole events highlighted as filled bars
  4. labeling_val_summary.csv / .json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_FOIL_ORDER = [0, 1, 2]
_CMAP = plt.get_cmap("viridis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotated labeling visuals with validation-hole marks")
    parser.add_argument("--data", required=True, help="Labelled Stage-2 CSV")
    parser.add_argument("--checkpoint", required=True, help="Path to best_finetune.pth")
    parser.add_argument("--split-summary", required=True, help="split_summary.json with holdout_holes")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--sieve-distance-cm", type=float, default=253.0)
    parser.add_argument("--max-points-per-foil", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_inference(checkpoint: str, data: str, device: str = "cpu") -> tuple[pd.DataFrame, np.ndarray]:
    import os, sys, torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from training.data.preprocessing import ScalerBundle
    from training.data.stage2_root_dataset import Stage2RootDataset
    from training.models import build_model_from_config
    from torch.utils.data import DataLoader

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = dict(cfg.get("model", {}))
    model = build_model_from_config(model_cfg, input_dim=int(model_cfg.get("input_dim", 5)))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    scaler_rel = cfg.get("pretrained", {}).get("scaler_bundle_path")
    scaler_path = os.path.join(os.path.dirname(checkpoint), "..", "..", scaler_rel) if not os.path.isabs(scaler_rel) else scaler_rel
    scaler_path = os.path.normpath(scaler_path)
    scaler = ScalerBundle.load(scaler_path)

    dcfg = cfg.get("data", {})
    ds = Stage2RootDataset(
        data_source=data,
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
    )

    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    blocks = []
    TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
    for batch in loader:
        inp = batch["inputs"].to(device)
        out = model(inp)
        blocks.append(torch.cat([out[k] for k in TARGET_KEYS], dim=1).detach().cpu().numpy())
    pred_scaled = np.concatenate(blocks, axis=0).astype(np.float64)
    pred_phys = scaler.inverse_transform_Y(pred_scaled)

    df = ds.df.reset_index(drop=True).copy()
    df["nn_xptar"] = pred_phys[:, 1]
    df["nn_yptar"] = pred_phys[:, 2]
    df["nn_ztar"] = pred_phys[:, 3]
    df["root_xptar"] = df["P_gtr_th"].to_numpy(dtype=np.float64)
    df["root_yptar"] = df["P_gtr_ph"].to_numpy(dtype=np.float64)
    df["root_ztar"] = df["P_react_z"].to_numpy(dtype=np.float64)
    df["nn_sieve_x_cm"] = df["nn_xptar"] * 253.0
    df["nn_sieve_y_cm"] = df["nn_yptar"] * 253.0
    # Use the project_to_sieve output (already computed in the labelled table)
    # instead of the simplified P_gtr_th/ph * 253 approximation.
    df["root_sieve_x_cm"] = df["sieve_x"].to_numpy(dtype=np.float64)
    df["root_sieve_y_cm"] = df["sieve_y"].to_numpy(dtype=np.float64)
    return df, pred_phys


def _subsample(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed).sort_index()


def _build_limits_sieve(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays]) if len(arrays) else np.array([0.0])
    lo, hi = np.quantile(all_vals, [0.002, 0.998])
    pad = max((hi - lo) * 0.06, 0.8)
    return (float(lo - pad), float(hi + pad)), (float(lo - pad), float(hi + pad))


def _design_grid(design_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.sort(np.unique(np.round(design_df["design_x_cm"].to_numpy(dtype=np.float64), 6)))
    y_vals = np.sort(np.unique(np.round(design_df["design_y_cm"].to_numpy(dtype=np.float64), 6)))
    if len(x_vals) >= 2:
        x_mid = 0.5 * (x_vals[:-1] + x_vals[1:])
        x_edges = np.concatenate(([x_vals[0] - (x_vals[1] - x_vals[0]) * 0.5], x_mid, [x_vals[-1] + (x_vals[-1] - x_vals[-2]) * 0.5]))
    else:
        x_edges = x_vals
    if len(y_vals) >= 2:
        y_mid = 0.5 * (y_vals[:-1] + y_vals[1:])
        y_edges = np.concatenate(([y_vals[0] - (y_vals[1] - y_vals[0]) * 0.5], y_mid, [y_vals[-1] + (y_vals[-1] - y_vals[-2]) * 0.5]))
    else:
        y_edges = y_vals
    return x_edges, y_edges, x_vals, y_vals


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_summary, "r", encoding="utf-8") as fh:
        split = json.load(fh)
    holdout_holes = set(int(h) for h in split.get("holdout_holes", []))

    df = pd.read_csv(data_path)
    df["is_val"] = df["hole_id"].apply(lambda h: int(h) in holdout_holes)
    df["hole_id"] = df["hole_id"].astype(int)
    df["foil_position"] = df["foil_position"].astype(int)
    df["cluster"] = df["cluster"].astype(int)

    sieve_distance_cm = float(args.sieve_distance_cm)
    sieve_design = _resolve_sieve_design(df, sieve_distance_cm)

    # --- load inference ---
    plot_df, _ = _load_inference(args.checkpoint, str(data_path))

    # ====================================================================
    # FIGURE 1: Labeling results with validation holes annotated
    # ====================================================================
    x_lim, y_lim = _build_limits_sieve(
        df["sieve_x"].to_numpy(dtype=np.float64),
        df["sieve_y"].to_numpy(dtype=np.float64),
        df["cluster_center_x"].to_numpy(dtype=np.float64),
        df["cluster_center_y"].to_numpy(dtype=np.float64),
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 17), constrained_layout=True, sharex=True, sharey=True)

    for row_idx, foil in enumerate(_FOIL_ORDER):
        dff = df.loc[df["foil_position"] == foil].copy()
        dff_plot = _subsample(dff, max_points=args.max_points_per_foil, seed=args.seed)
        dff_train = dff_plot.loc[~dff_plot["is_val"]]
        dff_val = dff_plot.loc[dff_plot["is_val"]]

        # Left: clustered events
        ax = axes[row_idx, 0]
        if not dff_train.empty:
            cv = dff_train["cluster"].to_numpy(dtype=np.int64)
            ax.scatter(dff_train["sieve_x"], dff_train["sieve_y"], c=np.mod(cv, 256) / 255.0, cmap=_CMAP,
                       s=5, alpha=0.7, linewidths=0, rasterized=True, label="train")
        if not dff_val.empty:
            ax.scatter(dff_val["sieve_x"], dff_val["sieve_y"], s=16, color="red",
                       alpha=0.9, linewidths=0, zorder=4, rasterized=True, label="val hole")
        centers = dff[["foil_position", "cluster", "cluster_center_x", "cluster_center_y"]].dropna().drop_duplicates(["cluster"])
        ax.scatter(centers["cluster_center_x"], centers["cluster_center_y"], marker="x", s=48, linewidths=1.4, color="black", zorder=5)
        ax.set_title(f"foil{foil} — clustered events (n_train={len(dff_train):,}, n_val={len(dff_val):,})")
        ax.grid(alpha=0.16); ax.set_aspect("equal"); ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
        if row_idx == 2: ax.set_xlabel("Sieve X [cm]")
        ax.set_ylabel("Sieve Y [cm]")
        ax.legend(loc="upper right", fontsize=7)

        # Right: cluster→label with validation marks
        ax = axes[row_idx, 1]
        ax.scatter(dff_train["sieve_x"], dff_train["sieve_y"], s=4, color="#c7c7c7", alpha=0.18, linewidths=0, rasterized=True)
        if not dff_val.empty:
            ax.scatter(dff_val["sieve_x"], dff_val["sieve_y"], s=14, color="red",
                       alpha=0.85, linewidths=0, zorder=4, rasterized=True)
        # unique cluster→label pairs
        mapping = dff[["cluster", "cluster_center_x", "cluster_center_y", "hole_row", "hole_col",
                        "matched_sieve_x_cm", "matched_sieve_y_cm"]].dropna().drop_duplicates("cluster")
        for row in mapping.itertuples(index=False):
            color = "red" if int(dff.loc[dff["cluster"] == row.cluster, "hole_id"].iloc[0]) in holdout_holes else "#2ca02c"
            lw = 2.0 if int(dff.loc[dff["cluster"] == row.cluster, "hole_id"].iloc[0]) in holdout_holes else 0.9
            ax.plot([row.cluster_center_x, row.matched_sieve_x_cm], [row.cluster_center_y, row.matched_sieve_y_cm],
                    color=color, alpha=0.6 if color != "red" else 0.9, linewidth=lw, zorder=2)
        ax.scatter(mapping["cluster_center_x"], mapping["cluster_center_y"], marker="x", s=46, linewidths=1.3, color="#1f77b4", zorder=4)
        ax.scatter(mapping["matched_sieve_x_cm"], mapping["matched_sieve_y_cm"], marker="o", s=54,
                   facecolors="none", edgecolors="#d62728", linewidths=1.2, zorder=5)
        ax.set_title(f"foil{foil} — cluster → label (val holes: red)")
        ax.grid(alpha=0.16); ax.set_aspect("equal"); ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
        if row_idx == 2: ax.set_xlabel("Sieve X [cm]")
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#c7c7c7", markeredgecolor="#c7c7c7", markersize=5, label="train events"),
            Line2D([0], [0], color="#2ca02c", linewidth=1.2, label="train match"),
            Line2D([0], [0], color="red", linewidth=2.0, label="val match"),
            Line2D([0], [0], marker="x", color="#1f77b4", linestyle="None", markersize=7, label="cluster center"),
            Line2D([0], [0], marker="o", color="#d62728", markerfacecolor="none", linestyle="None", markersize=7, label="label center"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7)

    fig.suptitle("Labeling results — validation holes highlighted", fontsize=16)
    fig.savefig(output_dir / "labeling_results_3x2_val_annotated.png", dpi=200)
    plt.close(fig)

    # ====================================================================
    # FIGURE 2: Sieve pattern with validation overlay
    # ====================================================================
    x_lim2, y_lim2 = _build_limits_sieve(
        plot_df["nn_sieve_x_cm"].to_numpy(dtype=np.float64),
        plot_df["nn_sieve_y_cm"].to_numpy(dtype=np.float64),
        plot_df["root_sieve_x_cm"].to_numpy(dtype=np.float64),
        plot_df["root_sieve_y_cm"].to_numpy(dtype=np.float64),
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 17), constrained_layout=True, sharex=True, sharey=True)

    for row_idx, foil in enumerate(_FOIL_ORDER):
        dff = plot_df.loc[plot_df["foil_position"] == foil].copy()
        dff_val = dff.loc[dff["hole_id"].apply(lambda h: int(h) in holdout_holes)]
        x_edges, y_edges, _, _ = _design_grid(sieve_design.loc[sieve_design["foil_position"] == foil])

        for col_idx, (method, cmap_name) in enumerate([("nn", "viridis"), ("root", "viridis")]):
            ax = axes[row_idx, col_idx]
            x = dff[f"{method}_sieve_x_cm"].to_numpy(dtype=np.float64)
            y = dff[f"{method}_sieve_y_cm"].to_numpy(dtype=np.float64)
            fm = np.isfinite(x) & np.isfinite(y)
            x, y = x[fm], y[fm]

            # grid lines
            for xe in x_edges: ax.axvline(xe, color="#d62728", linestyle="-", linewidth=0.5, alpha=0.3)
            for ye in y_edges: ax.axhline(ye, color="#d62728", linestyle="-", linewidth=0.5, alpha=0.3)

            # design-hole center crosses (1/4 grid spacing)
            foil_design = sieve_design.loc[sieve_design["foil_position"] == foil]
            if not foil_design.empty:
                x_vals_d = np.sort(np.unique(np.round(foil_design["design_x_cm"].to_numpy(dtype=np.float64), 6)))
                y_vals_d = np.sort(np.unique(np.round(foil_design["design_y_cm"].to_numpy(dtype=np.float64), 6)))
                hw_x = float(np.median(np.diff(x_vals_d)) / 8.0) if len(x_vals_d) >= 2 else 0.3125
                hw_y = float(np.median(np.diff(y_vals_d)) / 8.0) if len(y_vals_d) >= 2 else 0.205
                for xc in x_vals_d:
                    for yc in y_vals_d:
                        ax.plot([xc - hw_x, xc + hw_x], [yc, yc], color="#d62728", linewidth=0.75, alpha=0.65, zorder=8)
                        ax.plot([xc, xc], [yc - hw_y, yc + hw_y], color="#d62728", linewidth=0.75, alpha=0.65, zorder=8)

            hist = ax.hist2d(x, y, bins=200, range=[x_lim2, y_lim2], cmap=cmap_name, norm=LogNorm(vmin=1), cmin=1)

            # Validation events as a second hist2d layer (Reds)
            if not dff_val.empty:
                vx = dff_val[f"{method}_sieve_x_cm"].to_numpy(dtype=np.float64)
                vy = dff_val[f"{method}_sieve_y_cm"].to_numpy(dtype=np.float64)
                vfm = np.isfinite(vx) & np.isfinite(vy)
                if vfm.sum() > 0:
                    ax.hist2d(vx[vfm], vy[vfm], bins=200, range=[x_lim2, y_lim2],
                              cmap="Reds", norm=LogNorm(vmin=1), cmin=1, alpha=0.85, zorder=5)

            ax.set_title(f"foil{foil} — {method.upper()} reconstruction")
            ax.set_aspect("equal"); ax.set_xlim(*x_lim2); ax.set_ylim(*y_lim2); ax.grid(alpha=0.12)
            if row_idx == 2: ax.set_xlabel("Sieve X [cm]")
            if col_idx == 0: ax.set_ylabel("Sieve Y [cm]")
            if row_idx == 0 and col_idx == 1:
                ax.legend(handles=[
                    Line2D([0], [0], marker="s", color="none", markerfacecolor="red", markeredgecolor="red", markersize=8, alpha=0.7, label="val hole (Reds layer)"),
                ], loc="upper right", fontsize=8)

    fig.suptitle("Sieve-pattern reconstruction — validation holes highlighted", fontsize=16)
    fig.savefig(output_dir / "sieve_pattern_3x2_val_annotated.png", dpi=200)
    plt.close(fig)

    # ====================================================================
    # FIGURE 3: ztar distribution with validation events on independent y-axis
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panel_info = [("all", plot_df)] + [(f"foil{foil}", plot_df.loc[plot_df["foil_position"] == foil]) for foil in _FOIL_ORDER]

    for ax, (name, dff) in zip(axes.flatten(), panel_info):
        nn = dff["nn_ztar"].to_numpy(dtype=np.float64)
        root = dff["root_ztar"].to_numpy(dtype=np.float64)
        dff_val = dff.loc[dff["hole_id"].apply(lambda h: int(h) in holdout_holes)]

        finite = np.concatenate([nn[np.isfinite(nn)], root[np.isfinite(root)]])
        lo, hi = np.quantile(finite, [0.005, 0.995])
        pad = max((hi - lo) * 0.08, 0.5)
        bins = np.linspace(lo - pad, hi + pad, 70)

        ax.hist(root, bins=bins, density=True, alpha=0.4, color="#7f7f7f", label="ROOT ztar (train+val)", zorder=2)
        ax.hist(nn, bins=bins, density=True, alpha=0.4, color="#1f77b4", label="NN ztar (train+val)", zorder=2)

        # Validation events on a twin y-axis as filled histogram
        ax_val = ax.twinx()
        if not dff_val.empty:
            val_nn = dff_val["nn_ztar"].to_numpy(dtype=np.float64)
            val_root = dff_val["root_ztar"].to_numpy(dtype=np.float64)
            ax_val.hist(val_nn, bins=bins, density=True, alpha=0.55, color="red", label="NN val", zorder=3)
            ax_val.hist(val_root, bins=bins, density=True, alpha=0.35, color="darkorange", label="ROOT val", zorder=3)
        ax_val.set_ylabel("Val Density", color="red", fontsize=9)
        ax_val.tick_params(axis="y", labelcolor="red", labelsize=8)

        ax.set_title(f"{name}  (n_val={len(dff_val):,})")
        ax.set_xlabel("ztar [cm]")
        ax.set_ylabel("Density (train+val)", fontsize=9)
        ax.grid(alpha=0.15)

        # Combine legends from both axes
        lines_a, labels_a = ax.get_legend_handles_labels()
        lines_b, labels_b = ax_val.get_legend_handles_labels()
        ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc="upper right")

    fig.suptitle("ztar(vertex-z) distribution — validation events on independent axis", fontsize=16)
    fig.savefig(output_dir / "ztar_val_annotated.png", dpi=180)
    plt.close(fig)

    # ====================================================================
    # Summary
    # ====================================================================
    val_events = int(df["is_val"].sum())
    val_per_foil = {int(f): int((df["foil_position"] == f).sum()) for f in _FOIL_ORDER if (df["foil_position"] == f).any()}
    payload = {
        "holdout_holes": sorted(holdout_holes),
        "n_val_events": val_events,
        "n_train_events": int(len(df) - val_events),
        "val_events_per_foil": val_per_foil,
        "artifacts": {
            "labeling_plot": str(output_dir / "labeling_results_3x2_val_annotated.png"),
            "sieve_plot": str(output_dir / "sieve_pattern_3x2_val_annotated.png"),
            "ztar_plot": str(output_dir / "ztar_val_annotated.png"),
        },
    }
    with open(output_dir / "val_annotated_summary.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    pd.DataFrame([payload]).to_csv(output_dir / "val_annotated_summary.csv", index=False)

    print("Validation-annotated visuals complete")
    for _, p in payload["artifacts"].items():
        print(f"  {p}")


def _resolve_sieve_design(df: pd.DataFrame, sieve_distance_cm: float) -> pd.DataFrame:
    design_cols = ["foil_position", "hole_row", "hole_col"]
    if {"candidate_sieve_x_cm", "candidate_sieve_y_cm"}.issubset(df.columns):
        design = (df[design_cols + ["candidate_sieve_x_cm", "candidate_sieve_y_cm"]].dropna()
                  .groupby(design_cols, as_index=False)[["candidate_sieve_x_cm", "candidate_sieve_y_cm"]].median()
                  .rename(columns={"candidate_sieve_x_cm": "design_x_cm", "candidate_sieve_y_cm": "design_y_cm"}))
        return design
    if {"weak_hole_xptar_center", "weak_hole_yptar_center"}.issubset(df.columns):
        design = (df[design_cols + ["weak_hole_xptar_center", "weak_hole_yptar_center"]].dropna()
                  .groupby(design_cols, as_index=False)[["weak_hole_xptar_center", "weak_hole_yptar_center"]].median())
        design["design_x_cm"] = design["weak_hole_xptar_center"].to_numpy(dtype=np.float64) * sieve_distance_cm
        design["design_y_cm"] = design["weak_hole_yptar_center"].to_numpy(dtype=np.float64) * sieve_distance_cm
        return design[design_cols + ["design_x_cm", "design_y_cm"]]
    raise RuntimeError("Could not resolve sieve design centers.")


if __name__ == "__main__":
    main()
