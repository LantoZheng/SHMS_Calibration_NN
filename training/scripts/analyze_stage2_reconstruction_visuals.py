#!/usr/bin/env python3
"""Generate post-training reconstruction visuals for Stage-2 SHMS transport runs.

Outputs
-------
1. A 3x2 six-panel sieve-pattern matrix:
   rows   = foil0 / foil1 / foil2
   cols   = NN reconstruction / ROOT-HCANA reconstruction
2. A ztar(vertex-z) comparison figure against ROOT/HCANA.
3. JSON/CSV summaries for quick comparison across runs.

Notes
-----
- In the current Stage-2 experimental workflow, the network output key `ytar`
  corresponds physically to the reaction vertex z quantity compared to
  `P_react_z` in the labelled experimental table.
- Sieve-plane coordinates are computed from `(xptar, yptar)` using the nominal
  SHMS sieve distance, unless explicit sieve-plane columns are already present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.scripts.analyze_stage2_ytar_distribution import (
    build_dataset,
    load_checkpoint_bundle,
    predict_physical,
    resolve_device,
    resolve_path,
)

_SIEVE_DISTANCE_CM = 253.0
_FOIL_ORDER = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 post-training reconstruction visuals")
    parser.add_argument("--checkpoint", required=True, help="Path to stage2 best_finetune.pth")
    parser.add_argument("--data", required=True, help="Path to stage2 labelled CSV/Parquet/ROOT")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summaries")
    parser.add_argument("--scaler-bundle", default=None, help="Optional override for scaler bundle path")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (default: auto)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--sieve-distance-cm", type=float, default=_SIEVE_DISTANCE_CM, help="Distance from target to sieve plane")
    return parser.parse_args()


def robust_sigma(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    q16, q84 = np.quantile(arr, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _resolve_sieve_design(df: pd.DataFrame, sieve_distance_cm: float) -> pd.DataFrame:
    required_meta = ["foil_position", "hole_row", "hole_col"]
    for key in required_meta:
        if key not in df.columns:
            raise RuntimeError(f"Dataset is missing required metadata column '{key}' for sieve plotting.")

    design_cols = ["foil_position", "hole_row", "hole_col"]
    if {"candidate_sieve_x_cm", "candidate_sieve_y_cm"}.issubset(df.columns):
        design = (
            df[design_cols + ["candidate_sieve_x_cm", "candidate_sieve_y_cm"]]
            .dropna()
            .groupby(design_cols, as_index=False)[["candidate_sieve_x_cm", "candidate_sieve_y_cm"]]
            .median()
            .rename(columns={
                "candidate_sieve_x_cm": "design_x_cm",
                "candidate_sieve_y_cm": "design_y_cm",
            })
        )
        return design

    if {"weak_hole_xptar_center", "weak_hole_yptar_center"}.issubset(df.columns):
        design = (
            df[design_cols + ["weak_hole_xptar_center", "weak_hole_yptar_center"]]
            .dropna()
            .groupby(design_cols, as_index=False)[["weak_hole_xptar_center", "weak_hole_yptar_center"]]
            .median()
        )
        design["design_x_cm"] = design["weak_hole_xptar_center"].to_numpy(dtype=np.float64) * sieve_distance_cm
        design["design_y_cm"] = design["weak_hole_yptar_center"].to_numpy(dtype=np.float64) * sieve_distance_cm
        return design[design_cols + ["design_x_cm", "design_y_cm"]]

    raise RuntimeError("Could not resolve sieve-hole design centers from the dataset.")


def _build_plot_frame(df: pd.DataFrame, pred_phys: np.ndarray, sieve_distance_cm: float) -> pd.DataFrame:
    plot_df = df.reset_index(drop=True).copy()
    plot_df["nn_xptar"] = pred_phys[:, 1]
    plot_df["nn_yptar"] = pred_phys[:, 2]
    plot_df["nn_ztar"] = pred_phys[:, 3]
    plot_df["root_xptar"] = plot_df["P_gtr_th"].to_numpy(dtype=np.float64)
    plot_df["root_yptar"] = plot_df["P_gtr_ph"].to_numpy(dtype=np.float64)
    plot_df["root_ztar"] = plot_df["P_react_z"].to_numpy(dtype=np.float64)

    plot_df["nn_sieve_x_cm"] = plot_df["nn_xptar"].to_numpy(dtype=np.float64) * sieve_distance_cm
    plot_df["nn_sieve_y_cm"] = plot_df["nn_yptar"].to_numpy(dtype=np.float64) * sieve_distance_cm
    # Use the project_to_sieve output (already computed in the labelled table)
    # instead of the simplified P_gtr_th/ph * 253 approximation.
    plot_df["root_sieve_x_cm"] = plot_df["sieve_x"].to_numpy(dtype=np.float64)
    plot_df["root_sieve_y_cm"] = plot_df["sieve_y"].to_numpy(dtype=np.float64)

    foil_center_col = "weak_foil_ytar_center" if "weak_foil_ytar_center" in plot_df.columns else "foil_ytar_center"
    foil_tol_col = "weak_foil_ytar_tol" if "weak_foil_ytar_tol" in plot_df.columns else None
    plot_df["ztar_center_cm"] = plot_df[foil_center_col].to_numpy(dtype=np.float64)
    plot_df["ztar_tol_cm"] = (
        plot_df[foil_tol_col].to_numpy(dtype=np.float64)
        if foil_tol_col and foil_tol_col in plot_df.columns
        else np.zeros(len(plot_df), dtype=np.float64)
    )
    return plot_df


def _merge_design(plot_df: pd.DataFrame, design_df: pd.DataFrame) -> pd.DataFrame:
    merged = plot_df.merge(design_df, on=["foil_position", "hole_row", "hole_col"], how="left")
    if merged[["design_x_cm", "design_y_cm"]].isna().any().any():
        missing = merged.loc[merged["design_x_cm"].isna() | merged["design_y_cm"].isna(), ["foil_position", "hole_row", "hole_col"]].head(10)
        raise RuntimeError(
            "Sieve design merge left missing centers for some events, e.g. "
            + missing.to_dict(orient="records").__repr__()
        )
    return merged


def _describe_sieve_slice(dff: pd.DataFrame, method: str) -> Dict[str, float]:
    x_col = f"{method}_sieve_x_cm"
    y_col = f"{method}_sieve_y_cm"
    dx = dff[x_col].to_numpy(dtype=np.float64) - dff["design_x_cm"].to_numpy(dtype=np.float64)
    dy = dff[y_col].to_numpy(dtype=np.float64) - dff["design_y_cm"].to_numpy(dtype=np.float64)
    radial = np.sqrt(dx ** 2 + dy ** 2)
    return {
        "rmse_x_cm": rmse(dff[x_col].to_numpy(dtype=np.float64), dff["design_x_cm"].to_numpy(dtype=np.float64)),
        "rmse_y_cm": rmse(dff[y_col].to_numpy(dtype=np.float64), dff["design_y_cm"].to_numpy(dtype=np.float64)),
        "rmse_radial_cm": float(np.sqrt(np.mean(radial ** 2))),
        "sigma68_radial_cm": robust_sigma(radial),
        "bias_x_cm": float(np.mean(dx)),
        "bias_y_cm": float(np.mean(dy)),
    }


def _describe_ztar_slice(dff: pd.DataFrame) -> Dict[str, float]:
    center = dff["ztar_center_cm"].to_numpy(dtype=np.float64)
    root = dff["root_ztar"].to_numpy(dtype=np.float64)
    nn = dff["nn_ztar"].to_numpy(dtype=np.float64)
    nn_res = nn - center
    root_res = root - center
    tol = dff["ztar_tol_cm"].to_numpy(dtype=np.float64)
    return {
        "nn_rmse_cm": rmse(nn, center),
        "root_rmse_cm": rmse(root, center),
        "nn_sigma68_cm": robust_sigma(nn_res),
        "root_sigma68_cm": robust_sigma(root_res),
        "nn_bias_cm": float(np.mean(nn_res)),
        "root_bias_cm": float(np.mean(root_res)),
        "nn_within_tol": float(np.mean(np.abs(nn_res) <= tol)) if np.any(np.isfinite(tol)) else float("nan"),
        "root_within_tol": float(np.mean(np.abs(root_res) <= tol)) if np.any(np.isfinite(tol)) else float("nan"),
    }


def _plot_sieve_matrix(plot_df: pd.DataFrame, output_path: Path) -> list[Dict[str, Any]]:
    foil_values = [foil for foil in _FOIL_ORDER if foil in set(plot_df["foil_position"].dropna().astype(int).tolist())]
    if len(foil_values) != 3:
        foil_values = sorted(int(v) for v in plot_df["foil_position"].dropna().unique())

    finite_xy = np.concatenate(
        [
            plot_df[["nn_sieve_x_cm", "root_sieve_x_cm", "design_x_cm"]].to_numpy(dtype=np.float64).ravel(),
            plot_df[["nn_sieve_y_cm", "root_sieve_y_cm", "design_y_cm"]].to_numpy(dtype=np.float64).ravel(),
        ]
    )
    finite_xy = finite_xy[np.isfinite(finite_xy)]
    lo, hi = np.quantile(finite_xy, [0.002, 0.998])
    pad = max((hi - lo) * 0.05, 0.8)
    x_lim = (lo - pad, hi + pad)
    y_lim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(len(foil_values), 2, figsize=(13, 16), constrained_layout=True, sharex=True, sharey=True)
    if len(foil_values) == 1:
        axes = np.array([axes])

    summaries: list[Dict[str, Any]] = []
    methods = [
        ("nn", "NN reconstruction", "viridis"),
        ("root", "ROOT/HCANA reconstruction", "viridis"),
    ]

    def draw_design_crosses(ax, dff: pd.DataFrame) -> None:
        x_vals = np.sort(np.unique(np.round(dff["design_x_cm"].to_numpy(dtype=np.float64), 6)))
        y_vals = np.sort(np.unique(np.round(dff["design_y_cm"].to_numpy(dtype=np.float64), 6)))
        hw_x = float(np.median(np.diff(x_vals)) / 8.0) if len(x_vals) >= 2 else 0.3125
        hw_y = float(np.median(np.diff(y_vals)) / 8.0) if len(y_vals) >= 2 else 0.205
        for xc in x_vals:
            for yc in y_vals:
                ax.plot([xc - hw_x, xc + hw_x], [yc, yc], color="#d62728", linewidth=0.75, alpha=0.65, zorder=7)
                ax.plot([xc, xc], [yc - hw_y, yc + hw_y], color="#d62728", linewidth=0.75, alpha=0.65, zorder=7)

    def draw_design_grid(ax, dff: pd.DataFrame) -> None:
        x_vals = np.sort(np.unique(np.round(dff["design_x_cm"].to_numpy(dtype=np.float64), 6)))
        y_vals = np.sort(np.unique(np.round(dff["design_y_cm"].to_numpy(dtype=np.float64), 6)))
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

        for idx, x_edge in enumerate(x_edges):
            ax.axvline(
                x_edge,
                color="#d62728",
                linestyle="-",
                linewidth=0.6,
                alpha=0.35,
                zorder=0,
                label="sieve grid" if idx == 0 else None,
            )
        for y_edge in y_edges:
            ax.axhline(
                y_edge,
                color="#d62728",
                linestyle="-",
                linewidth=0.6,
                alpha=0.35,
                zorder=0,
            )

    for row_idx, foil in enumerate(foil_values):
        dff = plot_df.loc[plot_df["foil_position"] == foil].copy()
        for col_idx, (method, title, color) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            x = dff[f"{method}_sieve_x_cm"].to_numpy(dtype=np.float64)
            y = dff[f"{method}_sieve_y_cm"].to_numpy(dtype=np.float64)
            draw_design_grid(ax, dff)
            draw_design_crosses(ax, dff)
            finite_mask = np.isfinite(x) & np.isfinite(y)
            x = x[finite_mask]
            y = y[finite_mask]
            hist = ax.hist2d(
                x,
                y,
                bins=200,
                range=[x_lim, y_lim],
                cmap=color,
                norm=LogNorm(vmin=1),
                cmin=1,
            )
            stats = _describe_sieve_slice(dff, method)
            summaries.append({
                "foil": int(foil),
                "method": method,
                "n_events": int(len(dff)),
                **stats,
            })
            ax.set_title(
                f"foil{foil} — {title}\n"
                f"radial RMSE={stats['rmse_radial_cm']:.2f} cm, σ68={stats['sigma68_radial_cm']:.2f} cm"
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.grid(alpha=0.16)
            if row_idx == len(foil_values) - 1:
                ax.set_xlabel("Sieve X [cm]")
            if col_idx == 0:
                ax.set_ylabel("Sieve Y [cm]")
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right", fontsize=8)
            cbar = fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Counts", fontsize=8)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Stage-2 sieve-pattern reconstruction vs ROOT baseline", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summaries


def _plot_ztar_comparison(plot_df: pd.DataFrame, output_path: Path) -> list[Dict[str, Any]]:
    foil_values = [foil for foil in _FOIL_ORDER if foil in set(plot_df["foil_position"].dropna().astype(int).tolist())]
    if len(foil_values) != 3:
        foil_values = sorted(int(v) for v in plot_df["foil_position"].dropna().unique())

    summary_rows: list[Dict[str, Any]] = []
    summary_rows.append({"slice": "all", **_describe_ztar_slice(plot_df)})
    for foil in foil_values:
        dff = plot_df.loc[plot_df["foil_position"] == foil].copy()
        summary_rows.append({"slice": f"foil{foil}", **_describe_ztar_slice(dff)})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panel_axes = axes.flatten()
    panel_slices = [("all", plot_df)] + [(f"foil{foil}", plot_df.loc[plot_df["foil_position"] == foil].copy()) for foil in foil_values]

    for ax, (name, dff) in zip(panel_axes, panel_slices):
        nn = dff["nn_ztar"].to_numpy(dtype=np.float64)
        root = dff["root_ztar"].to_numpy(dtype=np.float64)
        center = dff["ztar_center_cm"].to_numpy(dtype=np.float64)
        finite = np.concatenate([nn[np.isfinite(nn)], root[np.isfinite(root)], center[np.isfinite(center)]])
        lo, hi = np.quantile(finite, [0.005, 0.995])
        pad = max((hi - lo) * 0.08, 0.5)
        bins = np.linspace(lo - pad, hi + pad, 80)
        ax.hist(root, bins=bins, density=True, alpha=0.45, color="#7f7f7f", label="ROOT/HCANA ztar")
        ax.hist(nn, bins=bins, density=True, alpha=0.45, color="#1f77b4", label="NN ztar")
        centers_unique = sorted({float(v) for v in np.round(center, 6)})
        for idx, c in enumerate(centers_unique):
            ax.axvline(c, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.75, label="foil center" if idx == 0 else None)
        stats = next(row for row in summary_rows if row["slice"] == name)
        ax.set_title(
            f"{name}: ROOT RMSE={stats['root_rmse_cm']:.3f} cm, NN RMSE={stats['nn_rmse_cm']:.3f} cm\n"
            f"ROOT σ68={stats['root_sigma68_cm']:.3f} cm, NN σ68={stats['nn_sigma68_cm']:.3f} cm"
        )
        ax.set_xlabel("ztar / vertex-z [cm]")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)

    fig.suptitle("Stage-2 ztar(vertex-z) comparison vs ROOT baseline", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary_rows


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def run_post_training_visuals(
    *,
    checkpoint: str | Path,
    data: str | Path,
    output_dir: str | Path,
    scaler_bundle: str | None = None,
    device: str | None = None,
    batch_size: int = 4096,
    max_events: int | None = None,
    sieve_distance_cm: float = _SIEVE_DISTANCE_CM,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = resolve_path(repo_root, str(checkpoint))
    data_path = resolve_path(repo_root, str(data))
    output_path = resolve_path(repo_root, str(output_dir))
    output_path.mkdir(parents=True, exist_ok=True)

    torch_device = resolve_device(device)
    ckpt, cfg, model, scaler_obj, scaler_path = load_checkpoint_bundle(
        repo_root, checkpoint_path, scaler_bundle, torch_device
    )
    dataset = build_dataset(data_path, cfg, scaler_obj, max_events)
    pred_phys = predict_physical(model, dataset, scaler_obj, torch_device, batch_size)

    plot_df = _build_plot_frame(dataset.df, pred_phys, sieve_distance_cm)
    design_df = _resolve_sieve_design(plot_df, sieve_distance_cm)
    plot_df = _merge_design(plot_df, design_df)

    sieve_plot = output_path / "sieve_pattern_foil_matrix_3x2.png"
    ztar_plot = output_path / "ztar_distribution_by_foil.png"
    sieve_summary = _plot_sieve_matrix(plot_df, sieve_plot)
    ztar_summary = _plot_ztar_comparison(plot_df, ztar_plot)

    pd.DataFrame(sieve_summary).to_csv(output_path / "sieve_pattern_summary.csv", index=False)
    pd.DataFrame(ztar_summary).to_csv(output_path / "ztar_summary.csv", index=False)

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if ckpt.get("epoch", None) is not None else None,
        "checkpoint_val_loss": float(ckpt.get("val_loss", float("nan"))) if ckpt.get("val_loss", None) is not None else None,
        "data": str(data_path),
        "scaler_bundle": str(scaler_path),
        "device": str(torch_device),
        "dataset_summary": {
            "raw_events": int(dataset.summary.raw_events),
            "kept_events": int(dataset.summary.kept_events),
            "cutflow": dataset.summary.cutflow,
        },
        "sieve_distance_cm": float(sieve_distance_cm),
        "sieve_summary": sieve_summary,
        "ztar_summary": ztar_summary,
        "artifacts": {
            "sieve_matrix_plot": str(sieve_plot),
            "ztar_plot": str(ztar_plot),
            "sieve_summary_csv": str(output_path / "sieve_pattern_summary.csv"),
            "ztar_summary_csv": str(output_path / "ztar_summary.csv"),
        },
    }
    _save_json(output_path / "reconstruction_visuals_summary.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_post_training_visuals(
        checkpoint=args.checkpoint,
        data=args.data,
        output_dir=args.output_dir,
        scaler_bundle=args.scaler_bundle,
        device=args.device,
        batch_size=args.batch_size,
        max_events=args.max_events,
        sieve_distance_cm=args.sieve_distance_cm,
    )
    print("Stage-2 reconstruction visuals complete")
    print(f"  checkpoint : {payload['checkpoint']}")
    print(f"  data       : {payload['data']}")
    print(f"  output dir : {Path(args.output_dir).resolve()}")
    print("  artifacts:")
    for _, artifact_path in payload["artifacts"].items():
        print(f"    - {artifact_path}")


if __name__ == "__main__":
    main()
