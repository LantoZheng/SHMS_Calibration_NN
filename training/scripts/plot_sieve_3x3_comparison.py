#!/usr/bin/env python3
"""3x3 sieve-pattern comparison: v3 NN vs v1 NN vs HCANA/ROOT."""

from __future__ import annotations

import os, sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.models import build_model_from_config

_FOIL_ORDER = [0, 1, 2]
_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
_SIEVE_DISTANCE_CM = 253.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-v1", required=True)
    p.add_argument("--checkpoint-v3", required=True)
    p.add_argument("--scaler-bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


@torch.no_grad()
def predict_sieve(checkpoint: str, data: str, scaler_path: str, device: str, batch_size: int) -> pd.DataFrame:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model_from_config(dict(cfg.get("model", {})), input_dim=5)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    scaler = ScalerBundle.load(scaler_path)
    dcfg = cfg.get("data", {})
    ds = Stage2RootDataset(
        data_source=data, tree_name=dcfg.get("tree_name","T"), scaler_bundle=scaler,
        feature_schema=dcfg.get("feature_schema",["x_fp","y_fp","xp_fp","yp_fp","fry"]),
        branch_map=dcfg.get("branch_map",{}), label_map=dcfg.get("label_map",{}),
        metadata_cols=dcfg.get("metadata_cols",{}), weight_col=dcfg.get("weight_col",None),
        fry_mode=dcfg.get("fry_mode","direct_or_proxy"),
        direct_fry_branch=dcfg.get("direct_fry_branch",None),
        fry_proxy_branches=dcfg.get("fry_proxy_branches",[]),
        cuts=dcfg.get("cuts",{}),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    blocks = []
    for batch in loader:
        out = model(batch["inputs"].to(device))
        blocks.append(torch.cat([out[k] for k in _TARGET_KEYS], dim=1).cpu().numpy())
    pred_phys = scaler.inverse_transform_Y(np.concatenate(blocks, axis=0).astype(np.float64))

    df = ds.df.reset_index(drop=True).copy()
    df["nn_sieve_x"] = pred_phys[:,1] * _SIEVE_DISTANCE_CM
    df["nn_sieve_y"] = pred_phys[:,2] * _SIEVE_DISTANCE_CM
    df["root_sieve_x"] = df["sieve_x"].to_numpy(dtype=np.float64)
    df["root_sieve_y"] = df["sieve_y"].to_numpy(dtype=np.float64)
    df["foil_position"] = df["foil_position"].astype(int)
    return df


def _build_limits(*dfs: pd.DataFrame) -> tuple:
    all_x, all_y = [], []
    for df in dfs:
        for c in ["nn_sieve_x","nn_sieve_y","root_sieve_x","root_sieve_y"]:
            if c in df.columns:
                v = df[c].to_numpy(dtype=np.float64)
                v = v[np.isfinite(v)]
                (all_y if "_y" in c else all_x).append(v)
    x_arr = np.concatenate(all_x) if all_x else np.array([0.0])
    y_arr = np.concatenate(all_y) if all_y else np.array([0.0])
    lo_x, hi_x = np.quantile(x_arr, [0.002, 0.998])
    lo_y, hi_y = np.quantile(y_arr, [0.002, 0.998])
    px = max((hi_x-lo_x)*0.06, 0.8)
    py = max((hi_y-lo_y)*0.06, 0.8)
    return (float(lo_x-px), float(hi_x+px)), (float(lo_y-py), float(hi_y+py))


def _design_grid_and_crosses(ax, design_df: pd.DataFrame):
    xs = np.sort(np.unique(np.round(design_df["design_x_cm"].to_numpy(dtype=np.float64), 6)))
    ys = np.sort(np.unique(np.round(design_df["design_y_cm"].to_numpy(dtype=np.float64), 6)))
    if len(xs)>=2:
        xm=0.5*(xs[:-1]+xs[1:]); xe=np.concatenate(([xs[0]-(xs[1]-xs[0])*0.5],xm,[xs[-1]+(xs[-1]-xs[-2])*0.5]))
    else: xe=xs
    if len(ys)>=2:
        ym=0.5*(ys[:-1]+ys[1:]); ye=np.concatenate(([ys[0]-(ys[1]-ys[0])*0.5],ym,[ys[-1]+(ys[-1]-ys[-2])*0.5]))
    else: ye=ys
    for x in xe: ax.axvline(x, color="#d62728", lw=0.5, alpha=0.3)
    for y in ye: ax.axhline(y, color="#d62728", lw=0.5, alpha=0.3)
    hx = float(np.median(np.diff(xs))/8.0) if len(xs)>=2 else 0.3125
    hy = float(np.median(np.diff(ys))/8.0) if len(ys)>=2 else 0.205
    for xc in xs:
        for yc in ys:
            ax.plot([xc-hx, xc+hx], [yc, yc], color="#d62728", lw=0.75, alpha=0.65, zorder=8)
            ax.plot([xc, xc], [yc-hy, yc+hy], color="#d62728", lw=0.75, alpha=0.65, zorder=8)


def _resolve_sieve_design(df: pd.DataFrame) -> pd.DataFrame:
    dc = ["foil_position","hole_row","hole_col"]
    if {"candidate_sieve_x_cm","candidate_sieve_y_cm"}.issubset(df.columns):
        return (df[dc+["candidate_sieve_x_cm","candidate_sieve_y_cm"]].dropna()
                .groupby(dc, as_index=False)[["candidate_sieve_x_cm","candidate_sieve_y_cm"]].median()
                .rename(columns={"candidate_sieve_x_cm":"design_x_cm","candidate_sieve_y_cm":"design_y_cm"}))
    if {"weak_hole_xptar_center","weak_hole_yptar_center"}.issubset(df.columns):
        d = (df[dc+["weak_hole_xptar_center","weak_hole_yptar_center"]].dropna()
             .groupby(dc, as_index=False)[["weak_hole_xptar_center","weak_hole_yptar_center"]].median())
        d["design_x_cm"] = d["weak_hole_xptar_center"].to_numpy(dtype=np.float64)*_SIEVE_DISTANCE_CM
        d["design_y_cm"] = d["weak_hole_yptar_center"].to_numpy(dtype=np.float64)*_SIEVE_DISTANCE_CM
        return d[dc+["design_x_cm","design_y_cm"]]
    raise RuntimeError("Cannot resolve sieve design.")


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df_v1 = predict_sieve(args.checkpoint_v1, args.data, args.scaler_bundle, args.device, args.batch_size)
    df_v3 = predict_sieve(args.checkpoint_v3, args.data, args.scaler_bundle, args.device, args.batch_size)

    design = _resolve_sieve_design(pd.read_csv(args.data))
    x_lim = (-18.0, 18.0)
    y_lim = (-10.0, 10.0)

    fig, axes = plt.subplots(3, 3, figsize=(18, 17), constrained_layout=True, sharex=True, sharey=True)

    col_labels = ["v3 NN (per-foil tol)", "v1 NN (no tol)", "HCANA (project_to_sieve)"]
    cmaps = ["viridis", "viridis", "viridis"]

    for row_idx, foil in enumerate(_FOIL_ORDER):
        dff_v3 = df_v3.loc[df_v3["foil_position"]==foil]
        dff_v1 = df_v1.loc[df_v1["foil_position"]==foil]
        foil_design = design.loc[design["foil_position"]==foil]

        datasets = [
            ("v3 NN", dff_v3, "nn"),
            ("v1 NN", dff_v1, "nn"),
            ("HCANA", dff_v1, "root"),
        ]
        for col_idx, (label, dff, src) in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            x = dff[f"{src}_sieve_x"].to_numpy(dtype=np.float64) if src=="nn" else dff["root_sieve_x"].to_numpy(dtype=np.float64)
            y = dff[f"{src}_sieve_y"].to_numpy(dtype=np.float64) if src=="nn" else dff["root_sieve_y"].to_numpy(dtype=np.float64)
            fm = np.isfinite(x) & np.isfinite(y); x = x[fm]; y = y[fm]
            _design_grid_and_crosses(ax, foil_design)
            ax.hist2d(x, y, bins=200, range=[x_lim, y_lim], cmap=cmaps[col_idx],
                       norm=LogNorm(vmin=1), cmin=1)

            # compute radial RMSE relative to design centers
            dx = x - foil_design["design_x_cm"].to_numpy(dtype=np.float64).mean()  # rough, per-event would be better
            # simpler: just report stats from the summary files  
            ax.set_title(f"foil{foil} — {label}")
            ax.set_aspect("equal"); ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
            ax.grid(alpha=0.10)
            if row_idx==2: ax.set_xlabel("Sieve X [cm]")
            if col_idx==0: ax.set_ylabel("Sieve Y [cm]")

    # Add column titles
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(f"foil0 — {label}", fontsize=11, fontweight="bold")

    fig.suptitle("Sieve-pattern comparison: v3 NN vs v1 NN vs HCANA", fontsize=16, y=1.01)
    fig.savefig(out / "sieve_3x3_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'sieve_3x3_comparison.png'}")


if __name__ == "__main__":
    main()
