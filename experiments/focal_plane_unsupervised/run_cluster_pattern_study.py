#!/usr/bin/env python3
"""Study how sieve-HDBSCAN clusters behave in focal-plane coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
READ = FP + ["foil_position", "cluster", "sieve_x", "sieve_y", "cluster_center_x", "cluster_center_y"]


def cluster_summary(frame: pd.DataFrame, foil: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    part = frame.loc[frame.foil_position == foil].copy()
    scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(part[FP])
    part["_row"] = np.arange(len(part))
    labels = np.sort(part.cluster.unique())
    rows = []
    for label in labels:
        mask = part.cluster.to_numpy() == label
        x = scaled[mask]
        center = np.median(x, axis=0)
        radius = float(np.median(np.linalg.norm(x - center, axis=1)))
        rows.append({
            "foil": foil, "cluster": int(label), "population": int(mask.sum()), "fp_radius": radius,
            "fp0": center[0], "fp1": center[1], "fp2": center[2], "fp3": center[3],
            "sieve_x": float(np.median(part.loc[mask, "cluster_center_x"])),
            "sieve_y": float(np.median(part.loc[mask, "cluster_center_y"])),
        })
    summary = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    fp_centers = summary[["fp0", "fp1", "fp2", "fp3"]].to_numpy()
    sieve_centers = summary[["sieve_x", "sieve_y"]].to_numpy()
    fp_dist = pairwise_distances(fp_centers)
    sieve_dist = pairwise_distances(sieve_centers)
    np.fill_diagonal(fp_dist, np.inf)
    np.fill_diagonal(sieve_dist, np.inf)
    fp_nn = np.argmin(fp_dist, axis=1)
    sieve_nn = np.argmin(sieve_dist, axis=1)
    summary["fp_nn_distance"] = fp_dist[np.arange(len(summary)), fp_nn]
    summary["sieve_nn_distance"] = sieve_dist[np.arange(len(summary)), sieve_nn]
    summary["fp_separation_ratio"] = summary.fp_nn_distance / (summary.fp_radius + summary.fp_radius.iloc[fp_nn].to_numpy() + 1e-9)
    summary["nearest_neighbor_same"] = fp_nn == sieve_nn
    # Compare four-nearest-centroid neighborhoods in FP and sieve, not cluster identities.
    fp_order = np.argsort(fp_dist, axis=1)[:, :4]
    sieve_order = np.argsort(sieve_dist, axis=1)[:, :4]
    summary["neighbor_jaccard4"] = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(fp_order, sieve_order)]
    # Event-level FP neighborhood purity with respect to the existing sieve cluster.
    knn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(scaled).kneighbors(return_distance=False)[:, 1:]
    local = (part.cluster.to_numpy()[:, None] == part.cluster.to_numpy()[knn]).mean(axis=1)
    part["fp_local_purity"] = local
    purity = part.groupby("cluster", as_index=False).fp_local_purity.mean()
    summary = summary.merge(purity, on="cluster", how="left")
    return summary, scaled, part.cluster.to_numpy(), part


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(path, usecols=READ).dropna()
    summaries, payload = [], {}
    for foil in sorted(frame.foil_position.unique()):
        summary, scaled, labels, part = cluster_summary(frame, int(foil))
        summaries.append(summary)
        payload[int(foil)] = (summary, scaled, labels, part)
    table = pd.concat(summaries, ignore_index=True)
    table.to_csv(out / "fp_pattern_by_sieve_cluster.csv", index=False)

    # A. Cluster-centroid geometry in each foil's FP PCA plane, coloured by sieve coordinate.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, foil in enumerate(sorted(payload)):
        summary, _, _, _ = payload[foil]
        emb = PCA(n_components=2).fit_transform(summary[["fp0", "fp1", "fp2", "fp3"]])
        for row, value in enumerate(("sieve_x", "sieve_y")):
            ax = axes[row, col]
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=summary[value], s=18 + 0.7 * np.sqrt(summary.population), cmap="viridis")
            ax.set(title=f"foil {foil}: FP centroid PCA, colour={value}", xlabel="centroid PC1", ylabel="centroid PC2")
            fig.colorbar(sc, ax=ax, label=value)
    fig.savefig(out / "04_cluster_centroid_fp_patterns.png", dpi=180)
    plt.close(fig)

    # B. FP separability and topology preservation relative to sieve geometry.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.scatterplot(data=table, x="sieve_nn_distance", y="fp_separation_ratio", hue="foil", size="fp_local_purity", sizes=(15, 110), ax=axes[0])
    axes[0].axhline(1, color="black", lw=1, alpha=.6)
    axes[0].set(title="FP separation of known sieve clusters", xlabel="nearest sieve-centroid distance", ylabel="FP nearest-centre / within-cluster radius")
    sns.boxplot(data=table, x="foil", y="neighbor_jaccard4", ax=axes[1])
    sns.stripplot(data=table, x="foil", y="neighbor_jaccard4", color="black", alpha=.35, size=3, ax=axes[1])
    axes[1].set(title="Does local sieve topology survive in FP?", xlabel="foil", ylabel="Jaccard overlap of four nearest cluster-centres")
    fig.savefig(out / "05_separability_and_topology.png", dpi=180)
    plt.close(fig)

    # C. Pattern gallery: three most and least FP-pure clusters per foil.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, foil in enumerate(sorted(payload)):
        summary, scaled, labels, part = payload[foil]
        for row, ascending in enumerate((False, True)):
            chosen = summary.sort_values("fp_local_purity", ascending=ascending).head(3).cluster.to_numpy()
            ax = axes[row, col]
            sample = np.linspace(0, len(part) - 1, min(5000, len(part)), dtype=int)
            ax.scatter(part.P_dc_x_fp.to_numpy()[sample], part.P_dc_xp_fp.to_numpy()[sample], s=1, c="lightgray", alpha=.22)
            for label in chosen:
                sub = part.loc[part.cluster == label]
                ax.scatter(sub.P_dc_x_fp, sub.P_dc_xp_fp, s=3, alpha=.7, label=f"c{label}")
            ax.set(title=f"foil {foil}: {'highest' if not ascending else 'lowest'} FP purity", xlabel="P_dc_x_fp", ylabel="P_dc_xp_fp")
            ax.legend(markerscale=2, fontsize=8)
    fig.savefig(out / "06_fp_pattern_gallery.png", dpi=180)
    plt.close(fig)

    report = {"per_foil": {}}
    for foil, group in table.groupby("foil"):
        report["per_foil"][str(int(foil))] = {
            "clusters": int(len(group)),
            "median_fp_separation_ratio": float(group.fp_separation_ratio.median()),
            "fraction_fp_separated_ratio_gt_1": float((group.fp_separation_ratio > 1).mean()),
            "median_fp_local_purity": float(group.fp_local_purity.median()),
            "nearest_neighbor_topology_match_fraction": float(group.nearest_neighbor_same.mean()),
            "median_neighbor_jaccard4": float(group.neighbor_jaccard4.median()),
        }
    report["interpretation"] = "Clusters and foil assignments are used only to observe the already-built sieve reference; all FP statistics describe their observed detector-space behaviour."
    (out / "fp_pattern_study_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
