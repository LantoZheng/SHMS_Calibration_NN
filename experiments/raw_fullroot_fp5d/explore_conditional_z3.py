"""Test a conditional focal-plane normal coordinate as a replacement z_3.

The reconstructed sieve coordinates supply a weak two-dimensional prior.  A
smooth polynomial map removes the FP5D change explained by (x_sieve,y_sieve).
The first principal component of its residual is the candidate transverse
coordinate z_3.  It is unsupervised with respect to cluster/foil labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FEATURES = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
PAIRS = [("S1", 219, 220, 0), ("S2", 15, 68, 1), ("S3", 2, 3, 2), ("S4", 45, 56, 2)]


def oriented_auc(values: np.ndarray, labels: np.ndarray) -> float:
    auc = roc_auc_score(labels, values)
    return float(max(auc, 1.0 - auc))


def dprime(values: np.ndarray, labels: np.ndarray) -> float:
    a, b = values[labels == 0], values[labels == 1]
    return float(abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)) + 1e-12))


def main() -> None:
    source = RESULTS / "raw_fullroot_flow_hdbscan_labels.csv"
    df = pd.read_csv(source)
    data = df.loc[df.flow_hdbscan_cluster >= 0].copy()
    xy = data[["sieve_x", "sieve_y"]].to_numpy()
    fp = RobustScaler(quantile_range=(5, 95)).fit_transform(data[FEATURES])

    # This deliberately uses no foil, hole, y_tar, or cluster label.
    conditional_mean = make_pipeline(
        PolynomialFeatures(degree=5, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1e-2),
    ).fit(xy, fp).predict(xy)
    residual = fp - conditional_mean
    pca = PCA(n_components=5, random_state=25521).fit(residual)
    z3 = pca.transform(residual)[:, 0]
    data["conditional_fp5d_z3"] = z3

    rows = []
    for tag, first, second, band in PAIRS:
        part = data.loc[data.flow_hdbscan_cluster.isin([first, second])]
        labels = (part.flow_hdbscan_cluster.to_numpy() == second).astype(int)
        for name, values in [("reconstructed_ytar", part["P.gtr.y"].to_numpy()), ("conditional_fp5d_z3", part["conditional_fp5d_z3"].to_numpy())]:
            rows.append({
                "pair": tag, "band": band, "clusters": f"{first} vs {second}", "coordinate": name,
                "auc": oriented_auc(values, labels), "fisher_dprime_1d": dprime(values, labels),
                "mean_first": float(values[labels == 0].mean()), "mean_second": float(values[labels == 1].mean()),
            })
    pd.DataFrame(rows).to_csv(RESULTS / "conditional_fp5d_z3_pair_metrics.csv", index=False)
    (RESULTS / "conditional_fp5d_z3_model.json").write_text(json.dumps({
        "method": "z3 = PC1[FP5D - E(FP5D | reconstructed sieve_x, sieve_y)]",
        "conditional_mean_model": "degree-5 polynomial ridge regression, no cluster/foil/ytar labels",
        "residual_pc_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pc1_weights_in_robust_scaled_fp5d": dict(zip(FEATURES, pca.components_[0].tolist())),
    }, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 4, figsize=(21, 12), constrained_layout=True)
    colors = ["tab:blue", "tab:orange"]
    for col, (tag, first, second, band) in enumerate(PAIRS):
        part = data.loc[data.flow_hdbscan_cluster.isin([first, second])]
        labels = (part.flow_hdbscan_cluster.to_numpy() == second).astype(int)
        center = part.groupby("flow_hdbscan_cluster")[["sieve_x", "sieve_y"]].mean()
        dist = float(np.linalg.norm(center.loc[first] - center.loc[second]))
        for value, cluster, color in zip((0, 1), (first, second), colors):
            mask = labels == value
            axes[0, col].scatter(part.loc[mask, "sieve_x"], part.loc[mask, "sieve_y"], s=10, alpha=.75, color=color, label=str(cluster), linewidths=0)
            axes[0, col].scatter(center.loc[cluster, "sieve_x"], center.loc[cluster, "sieve_y"], marker="*", s=80, color="red", edgecolor="white", linewidth=.4)
        mid = center.mean(axis=0)
        span = max(.9, dist * 1.9, np.ptp(part[["sieve_x", "sieve_y"]].to_numpy(), axis=0).max()*.58)
        axes[0, col].set(xlim=(mid.sieve_x-span, mid.sieve_x+span), ylim=(mid.sieve_y-span, mid.sieve_y+span), xlabel=r"reconstructed $x_{sieve}$")
        axes[0, col].set_title(f"{tag}: {first} vs {second}, band {band}\ncentre distance={dist:.3f} cm")
        axes[0, col].grid(alpha=.16); axes[0, col].legend(frameon=False, fontsize=8)
        if col == 0: axes[0, col].set_ylabel(r"reconstructed $y_{sieve}$")

        for row, field, label in [(1, "P.gtr.y", r"reconstructed $y_{tar}$"), (2, "conditional_fp5d_z3", r"conditional FP5D $z_3$")]:
            for value, cluster, color in zip((0, 1), (first, second), colors):
                vals = part.loc[labels == value, field].to_numpy()
                bins = np.linspace(vals.min(), vals.max(), 30)
                axes[row, col].hist(vals, bins=bins, histtype="stepfilled", alpha=.35, color=color, label=str(cluster))
            metric = next(item for item in rows if item["pair"] == tag and item["coordinate"] == ("reconstructed_ytar" if row == 1 else "conditional_fp5d_z3"))
            axes[row, col].set_title(f"{label}: AUC={metric['auc']:.3f}; d'={metric['fisher_dprime_1d']:.2f}")
            axes[row, col].set_xlabel(label); axes[row, col].grid(alpha=.16)
            if col == 0: axes[row, col].set_ylabel("events / bin")

    fig.suptitle("Testing a conditional FP5D normal coordinate as a replacement for unreliable reconstructed $y_{tar}$", fontsize=17, fontweight="bold")
    fig.savefig(RESULTS / "conditional_fp5d_z3_same_band_near_pairs.png", dpi=200)


if __name__ == "__main__":
    main()
