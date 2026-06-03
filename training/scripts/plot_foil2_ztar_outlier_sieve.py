#!/usr/bin/env python3
"""Foil2 outlier diagnostic: events with P_react_z < -12 cm, NN vs HCANA sieve."""

import os, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.models import build_model_from_config

_SIEVE_DISTANCE_CM = 253.0
_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scaler-bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ztar-cut", type=float, default=-12.0)
    return p.parse_args()


@torch.no_grad()
def predict(checkpoint, data, scaler_path, device):
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
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    blocks = []
    for batch in loader:
        out = model(batch["inputs"].to(device))
        blocks.append(torch.cat([out[k] for k in _TARGET_KEYS], dim=1).cpu().numpy())
    pred_phys = scaler.inverse_transform_Y(np.concatenate(blocks, axis=0).astype(np.float64))

    df = ds.df.reset_index(drop=True).copy()
    df["nn_sieve_x"] = pred_phys[:,1] * _SIEVE_DISTANCE_CM
    df["nn_sieve_y"] = pred_phys[:,2] * _SIEVE_DISTANCE_CM
    df["nn_ytar"] = pred_phys[:,3]
    df["root_sieve_x"] = df["sieve_x"].to_numpy(dtype=np.float64)
    df["root_sieve_y"] = df["sieve_y"].to_numpy(dtype=np.float64)
    return df


def _design_grid_and_crosses(ax, design_df):
    xs = np.sort(np.unique(np.round(design_df["design_x_cm"].to_numpy(dtype=np.float64), 6)))
    ys = np.sort(np.unique(np.round(design_df["design_y_cm"].to_numpy(dtype=np.float64), 6)))
    if len(xs)>=2:
        xm=0.5*(xs[:-1]+xs[1:]); xe=np.concatenate(([xs[0]-(xs[1]-xs[0])*0.5],xm,[xs[-1]+(xs[-1]-xs[-2])*0.5]))
    else: xe=xs
    if len(ys)>=2:
        ym=0.5*(ys[:-1]+ys[1:]); ye=np.concatenate(([ys[0]-(ys[1]-ys[0])*0.5],ym,[ys[-1]+(ys[-1]-ys[-2])*0.5]))
    else: ye=ys
    for x in xe: ax.axvline(x, color="#d62728", lw=0.5, alpha=0.35)
    for y in ye: ax.axhline(y, color="#d62728", lw=0.5, alpha=0.35)
    hx = float(np.median(np.diff(xs))/8.0) if len(xs)>=2 else 0.3125
    hy = float(np.median(np.diff(ys))/8.0) if len(ys)>=2 else 0.205
    for xc in xs:
        for yc in ys:
            ax.plot([xc-hx, xc+hx], [yc, yc], color="#d62728", lw=0.75, alpha=0.65, zorder=8)
            ax.plot([xc, xc], [yc-hy, yc+hy], color="#d62728", lw=0.75, alpha=0.65, zorder=8)


def _resolve_design(df):
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
    raise RuntimeError("Cannot resolve design.")


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = predict(args.checkpoint, args.data, args.scaler_bundle, args.device)
    design = _resolve_design(pd.read_csv(args.data))

    # Filter: foil2 + P_react_z < ztar_cut
    dff = df.loc[(df["foil_position"]==2) & (df["P_react_z"] < args.ztar_cut)].copy()
    print(f"foil2 events with P_react_z < {args.ztar_cut}: {len(dff)}")
    if len(dff) == 0:
        print("No events match the cut.")
        return

    foil_design = design.loc[design["foil_position"]==2]
    x_lim, y_lim = (-18.0, 18.0), (-10.0, 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True, sharex=True, sharey=True)

    for col_idx, (label, x_col, y_col) in enumerate([
        ("NN (v1)", "nn_sieve_x", "nn_sieve_y"),
        ("HCANA", "root_sieve_x", "root_sieve_y"),
    ]):
        ax = axes[col_idx]
        x = dff[x_col].to_numpy(dtype=np.float64)
        y = dff[y_col].to_numpy(dtype=np.float64)
        fm = np.isfinite(x) & np.isfinite(y); x = x[fm]; y = y[fm]
        _design_grid_and_crosses(ax, foil_design)
        ax.hist2d(x, y, bins=200, range=[x_lim, y_lim], cmap="viridis",
                   norm=LogNorm(vmin=1), cmin=1)
        ax.set_title(f"foil2, P_react_z < {args.ztar_cut} cm — {label}  (n={len(x)})")
        ax.set_aspect("equal"); ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)
        ax.grid(alpha=0.10)
        ax.set_xlabel("Sieve X [cm]")
    axes[0].set_ylabel("Sieve Y [cm]")

    fig.suptitle(f"Foil2 extreme ztar events: P_react_z < {args.ztar_cut} cm", fontsize=15)
    fig.savefig(out / f"foil2_ztar_lt_{abs(args.ztar_cut):.0f}_nn_vs_hcana.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: ztar distributions ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    root_ztar = dff["P_react_z"].to_numpy(dtype=np.float64)
    nn_ytar = dff["nn_ytar"].to_numpy(dtype=np.float64)
    finite = np.concatenate([root_ztar[np.isfinite(root_ztar)], nn_ytar[np.isfinite(nn_ytar)]])
    lo, hi = np.quantile(finite, [0.005, 0.995])
    pad = max((hi-lo)*0.08, 0.5)
    bins = np.linspace(lo-pad, hi+pad, 50)
    ax2.hist(root_ztar, bins=bins, density=True, alpha=0.5, color="#d62728", label=f"HCANA P_react_z (mean={np.mean(root_ztar):.2f})")
    ax2.hist(nn_ytar, bins=bins, density=True, alpha=0.5, color="#1f77b4", label=f"NN ytar (mean={np.mean(nn_ytar):.2f})")
    ax2.axvline(-10.0, color="black", linestyle="--", linewidth=1.5, label="foil2 center (-10 cm)")
    ax2.axvline(args.ztar_cut, color="gray", linestyle=":", linewidth=1.2, label=f"ztar cut ({args.ztar_cut} cm)")
    ax2.set_xlabel("ztar [cm]"); ax2.set_ylabel("Density")
    ax2.set_title(f"Foil2 ztar distribution (P_react_z < {args.ztar_cut} cm, n={len(dff)})")
    ax2.legend(fontsize=10); ax2.grid(alpha=0.15)
    fig2.savefig(out / f"foil2_ztar_lt_{abs(args.ztar_cut):.0f}_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Also print ztar stats for these events
    root_ztar = dff["P_react_z"].to_numpy(dtype=np.float64)
    nn_ytar = dff["nn_ytar"].to_numpy(dtype=np.float64)
    print(f"  root_ztar: mean={np.mean(root_ztar):.2f}, std={np.std(root_ztar):.2f}, min={np.min(root_ztar):.2f}")
    print(f"  nn_ytar:   mean={np.mean(nn_ytar):.2f}, std={np.std(nn_ytar):.2f}, min={np.min(nn_ytar):.2f}")
    print(f"Saved: {out / f'foil2_ztar_lt_{abs(args.ztar_cut):.0f}_nn_vs_hcana.png'}")
    print(f"Saved: {out / f'foil2_ztar_lt_{abs(args.ztar_cut):.0f}_distribution.png'}")


if __name__ == "__main__":
    main()
