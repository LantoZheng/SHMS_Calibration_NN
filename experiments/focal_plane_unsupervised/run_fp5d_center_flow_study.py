"""Describe the raw 5D focal-plane flow traced by existing sieve-hole centers.

Existing sieve clustering supplies the reference grouping only.  Each output
point is the event-average of one (foil, reference-hole) group in measured FP
coordinates; no learned transform or dimensionality reduction is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
SIEVE = ["cluster_center_x", "cluster_center_y"]


def link_grid(ax, d, x, y, group):
    """Draw grid-neighbour trajectories, preserving measured coordinate axes."""
    for _, g in d.groupby(group, observed=True):
        g = g.sort_values("hole_col" if group == "hole_row" else "hole_row")
        if len(g) > 1:
            ax.plot(g[x], g[y], color="0.55", alpha=.28, lw=.65, zorder=1)


def main():
    root = Path(__file__).resolve().parents[2]
    data = root / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).parent / "results"; out.mkdir(exist_ok=True)
    cols = FP + SIEVE + ["foil_position", "cluster", "hole_row", "hole_col"]
    raw = pd.read_csv(data, usecols=cols).dropna()
    centers = raw.groupby(["foil_position", "cluster"], observed=True).agg(
        **{f"{c}_mean": (c, "mean") for c in FP},
        **{f"{c}_std": (c, "std") for c in FP},
        cluster_center_x=("cluster_center_x", "first"),
        cluster_center_y=("cluster_center_y", "first"),
        hole_row=("hole_row", "first"), hole_col=("hole_col", "first"), events=("cluster", "size"),
    ).reset_index()
    centers.to_csv(out / "fp5d_sieve_reference_centers.csv", index=False)
    fc = [f"{c}_mean" for c in FP]
    z = RobustScaler(quantile_range=(5,95)).fit_transform(centers[fc])
    summaries = []
    for foil, d in centers.groupby("foil_position", observed=True):
        ids = d.index.to_numpy()
        zz = z[ids]
        eig = np.linalg.eigvalsh(np.cov(zz, rowvar=False))[::-1]
        ratio = eig / eig.sum()
        fpdist = pdist(zz)
        sd = pdist(d[SIEVE])
        # The nearest *other* center in 5D exposes whether a sheet folds onto itself.
        dm = squareform(fpdist); np.fill_diagonal(dm, np.inf)
        nearest = dm.argmin(axis=1)
        grid_delta = np.linalg.norm(d[["hole_row", "hole_col"]].to_numpy() - d[["hole_row", "hole_col"]].to_numpy()[nearest], axis=1)
        summaries.append({"foil": int(foil), "holes": int(len(d)),
                          "center_pca_first2_fraction": float(ratio[:2].sum()),
                          "center_pca_effective_dimension": float((eig.sum()**2)/(eig**2).sum()),
                          "fp5d_vs_sieve_distance_spearman": float(spearmanr(fpdist, sd).statistic),
                          "median_5d_nearest_center_grid_separation": float(np.median(grid_delta)),
                          "fraction_5d_nearest_center_grid_adjacent": float((grid_delta <= np.sqrt(2)).mean())})
    # Compare cross-foil centre sheets at the same mechanical grid coordinate.
    cross = []
    for a in sorted(centers.foil_position.unique()):
        for b in sorted(centers.foil_position.unique()):
            if b <= a: continue
            m = centers[centers.foil_position==a].merge(centers[centers.foil_position==b], on=["hole_row","hole_col"], suffixes=("_a","_b"))
            arr_a = m[[f"{c}_mean_a" for c in FP]].to_numpy()
            arr_b = m[[f"{c}_mean_b" for c in FP]].to_numpy()
            scale = RobustScaler(quantile_range=(5,95)).fit(centers[fc].to_numpy())
            cross.append({"foil_pair": f"{int(a)}-{int(b)}", "common_grid_positions": int(len(m)),
                          "median_same_grid_5d_distance": float(np.median(np.linalg.norm(scale.transform(arr_a)-scale.transform(arr_b), axis=1)))})
    report = {"reference_centers": int(len(centers)), "fp_features": FP,
              "interpretation": "Each point is an existing sieve-cluster event mean in original measured FP coordinates.",
              "per_foil": summaries, "cross_foil_same_grid": cross}
    (out / "fp5d_center_flow_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Direct coordinate views: rows are foils; grey polylines connect adjacent
    # physical sieve positions and colours retain continuous sieve coordinates.
    views = [("P_dc_x_fp_mean", "P_dc_xp_fp_mean", "cluster_center_x", "xfp", "xpfp", "sieve x"),
             ("P_dc_y_fp_mean", "P_dc_yp_fp_mean", "cluster_center_y", "yfp", "ypfp", "sieve y"),
             ("P_dc_x_fp_mean", "P_rb_raster_frybRawAdc_mean", "cluster_center_y", "xfp", "fr_ybpm", "sieve y")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row, foil in enumerate(sorted(centers.foil_position.unique())):
        d = centers[centers.foil_position == foil]
        for col, (xname, yname, color, xl, yl, cl) in enumerate(views):
            ax = axes[row, col]
            link_grid(ax, d, xname, yname, "hole_row")
            link_grid(ax, d, xname, yname, "hole_col")
            im = ax.scatter(d[xname], d[yname], c=d[color], cmap="viridis", s=36, edgecolor="0.15", linewidth=.25, zorder=2)
            ax.set(title=f"foil {int(foil)}: {xl}–{yl}", xlabel=xl, ylabel=yl)
            if row == 0: fig.colorbar(im, ax=ax, label=cl)
    fig.savefig(out / "12_fp5d_reference_center_coordinate_flows.png", dpi=200)
    plt.close(fig)

    # Five dimensions shown simultaneously as original-coordinate centre profiles.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True, sharey=True)
    qlo, qhi = np.quantile(centers[fc].to_numpy(), [.02, .98], axis=0)
    scaled = (centers[fc].to_numpy() - qlo) / (qhi-qlo)
    for ax, foil in zip(axes, sorted(centers.foil_position.unique())):
        mask = centers.foil_position.to_numpy() == foil
        for line, c in zip(scaled[mask], centers.loc[mask, "cluster_center_y"]):
            ax.plot(range(5), line, color=plt.colormaps["plasma"]((c-centers.cluster_center_y.min())/(centers.cluster_center_y.max()-centers.cluster_center_y.min())), alpha=.5, lw=.8)
        ax.set(title=f"foil {int(foil)}", xticks=range(5), xticklabels=["xfp","yfp","xpfp","ypfp","fr_ybpm"], ylim=(-.08,1.08), ylabel="per-coordinate 2–98% range")
    fig.savefig(out / "13_fp5d_reference_center_profiles.png", dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
