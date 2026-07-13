#!/usr/bin/env python3
"""Create label-free multi-view diagnostics for focal-plane clustering."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
SIEVE = ["sieve_x", "sieve_y"]
ALL = FP + SIEVE


def load(n: int = 30000, seed: int = 25521) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    frame = pd.read_csv(path, usecols=ALL).dropna()
    rng = np.random.default_rng(seed)
    return frame.iloc[rng.choice(len(frame), min(n, len(frame)), replace=False)].reset_index(drop=True)


def hex(ax, x, y, xlabel, ylabel, gridsize=70):
    layer = ax.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=1, cmap="viridis")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return layer


def main() -> None:
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df = load()
    fp_scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP])
    sieve_scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(df[SIEVE])
    pca = PCA(n_components=4, random_state=25521).fit_transform(fp_scaled)
    corr = df[ALL].corr(method="spearman")

    # 1. Raw detector and weak-prior projections.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    pairs = [
        ("P_dc_x_fp", "P_dc_y_fp", "FP position plane"),
        ("P_dc_xp_fp", "P_dc_yp_fp", "FP slope plane"),
        ("P_dc_x_fp", "P_dc_xp_fp", "x position-slope coupling"),
        ("P_dc_y_fp", "P_dc_yp_fp", "y position-slope coupling"),
        ("sieve_x", "sieve_y", "current reconstructed sieve view"),
    ]
    for ax, (x, y, title) in zip(axes.flat, pairs):
        im = hex(ax, df[x], df[y], x, y)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="log10(count)")
    im = hex(axes.flat[-1], pca[:, 0], pca[:, 1], "FP PC1", "FP PC2")
    axes.flat[-1].set_title("FP manifold projection")
    fig.colorbar(im, ax=axes.flat[-1], label="log10(count)")
    fig.suptitle("25521: label-free focal-plane and sieve-prior projections", fontsize=14)
    fig.savefig(out / "01_multiview_density.png", dpi=180)
    plt.close(fig)

    # 2. Relationships and effective dimensionality.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", center=0, square=True, annot=True, fmt=".2f", ax=axes[0])
    axes[0].set_title("Spearman correlation: FP and sieve prior")
    variance = PCA(n_components=4).fit(fp_scaled).explained_variance_ratio_
    axes[1].bar(np.arange(1, 5), variance, color="#4C78A8")
    axes[1].plot(np.arange(1, 5), np.cumsum(variance), marker="o", color="#F58518", label="cumulative")
    axes[1].set(xticks=np.arange(1, 5), xlabel="FP principal component", ylabel="explained variance ratio", ylim=(0, 1.05))
    axes[1].legend()
    axes[1].set_title("Effective dimensionality of focal-plane cloud")
    fig.savefig(out / "02_dependence_and_dimension.png", dpi=180)
    plt.close(fig)

    # 3. Does a FP-local neighborhood remain sieve-local?
    nn = NearestNeighbors(n_neighbors=31, n_jobs=-1).fit(fp_scaled)
    fp_dist, indices = nn.kneighbors(fp_scaled)
    source = np.repeat(np.arange(len(df)), 30)
    neighbor = indices[:, 1:].ravel()
    sieve_dist = np.linalg.norm(sieve_scaled[source] - sieve_scaled[neighbor], axis=1)
    fp_edge = fp_dist[:, 1:].ravel()
    ranks = np.tile(np.arange(1, 31), len(df))
    bins = pd.qcut(fp_edge, q=20, duplicates="drop")
    edge_summary = pd.DataFrame({"fp_distance": fp_edge, "sieve_distance": sieve_dist, "rank": ranks, "bin": bins}).groupby("bin", observed=True).agg(
        fp_distance=("fp_distance", "median"), sieve_median=("sieve_distance", "median"), sieve_q90=("sieve_distance", lambda x: np.quantile(x, .9))
    ).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].scatter(fp_edge[::15], sieve_dist[::15], s=2, alpha=.12)
    axes[0].plot(edge_summary.fp_distance, edge_summary.sieve_median, color="#E45756", lw=2, label="median")
    axes[0].plot(edge_summary.fp_distance, edge_summary.sieve_q90, color="#72B7B2", lw=2, label="90th percentile")
    axes[0].set(xlabel="FP kNN edge distance (robust-scaled)", ylabel="sieve distance (robust-scaled)", title="Agreement of local neighborhoods")
    axes[0].legend()
    axes[1].hist(fp_dist[:, 20], bins=80, density=True, alpha=.8, label="20th FP neighbor")
    axes[1].hist(sieve_dist, bins=80, density=True, alpha=.55, label="sieve distance along FP edges")
    axes[1].set(xlabel="distance", ylabel="density", title="Density scale and prior mismatch")
    axes[1].legend()
    fig.savefig(out / "03_neighborhood_agreement.png", dpi=180)
    plt.close(fig)

    summary = {
        "n_events": len(df), "read_columns": ALL,
        "not_read": ["hole_id", "hole_row", "hole_col", "foil_position", "P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_gtr_y"],
        "fp_pca_variance": variance.tolist(),
        "fp_pc12_cumulative_variance": float(variance[:2].sum()),
        "median_fp_20th_neighbor_distance": float(np.median(fp_dist[:, 20])),
        "median_sieve_distance_on_fp_edges": float(np.median(sieve_dist)),
        "p90_sieve_distance_on_fp_edges": float(np.quantile(sieve_dist, .9)),
        "plots": ["01_multiview_density.png", "02_dependence_and_dimension.png", "03_neighborhood_agreement.png"],
    }
    (out / "data_observatory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
