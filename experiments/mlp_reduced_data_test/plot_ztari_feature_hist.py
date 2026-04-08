#!/usr/bin/env python3
"""Plot ztari versus all core features for passed events (stop_id == 0)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


def detect_prefix(branches: set[str]) -> str:
    if {"psxfp", "psztari"}.issubset(branches):
        return "ps"
    if {"hsxfp", "hsztari"}.issubset(branches):
        return "hs"
    raise RuntimeError("Cannot detect branch prefix. Expected ps* or hs* branches.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ztari vs all features for passed events")
    parser.add_argument("--root", required=True, help="Path to ROOT file")
    parser.add_argument(
        "--outdir",
        default="/Users/zhengxiaoyang/Desktop/AI_ML R-SIDIS/SHMS_Calibration_NN/experiments/mlp_reduced_data_test/outputs_notebook",
        help="Output directory for generated figures",
    )
    parser.add_argument("--tree", default="h10", help="TTree name")
    parser.add_argument("--bins", type=int, default=80, help="2D histogram bins")
    args = parser.parse_args()

    root_path = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with uproot.open(root_path) as f:
        tree = f[args.tree]
        branches = set(tree.keys())
        prefix = detect_prefix(branches)

        feature_names = [f"{prefix}xfp", f"{prefix}yfp", f"{prefix}xpfp", f"{prefix}ypfp", f"{prefix}deltai", f"{prefix}xptari", f"{prefix}yptari"]
        z_name = f"{prefix}ztari"

        required = feature_names + [z_name, "stop_id"]
        missing = [b for b in required if b not in branches]
        if missing:
            raise RuntimeError(f"Missing required branches: {missing}")

        arr = tree.arrays(required, library="np")

    mask = arr["stop_id"] == 0
    n_all = int(arr["stop_id"].shape[0])
    n_pass = int(mask.sum())
    if n_pass == 0:
        raise RuntimeError("No passed events found (stop_id == 0).")

    z = arr[z_name][mask]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()

    for i, feat in enumerate(feature_names):
        x = arr[feat][mask]
        h = axes[i].hist2d(x, z, bins=args.bins, cmap="viridis")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(z_name)
        axes[i].set_title(f"{z_name} vs {feat}")
        cb = fig.colorbar(h[3], ax=axes[i])
        cb.set_label("counts")

    # Last panel: ztari 1D distribution for passed events
    axes[-1].hist(z, bins=args.bins, color="#3b82f6", alpha=0.9)
    axes[-1].set_xlabel(z_name)
    axes[-1].set_ylabel("counts")
    axes[-1].set_title(f"{z_name} (passed only)")

    fig.suptitle(
        f"Passed events only (stop_id==0): {n_pass}/{n_all} ({n_pass/n_all:.1%}), prefix={prefix}",
        fontsize=14,
    )

    out_png = outdir / f"ztari_vs_features_passed_{root_path.stem}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    # Also save a compact summary text
    summary_txt = outdir / f"ztari_vs_features_passed_{root_path.stem}.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"root: {root_path}",
                f"tree: {args.tree}",
                f"prefix: {prefix}",
                f"total_events: {n_all}",
                f"passed_events: {n_pass}",
                f"passed_fraction: {n_pass/n_all:.6f}",
                f"ztari_branch: {z_name}",
                f"features: {', '.join(feature_names)}",
                f"output_png: {out_png}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved figure: {out_png}")
    print(f"Saved summary: {summary_txt}")


if __name__ == "__main__":
    main()
