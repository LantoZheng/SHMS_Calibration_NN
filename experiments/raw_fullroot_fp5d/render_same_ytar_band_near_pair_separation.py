"""Diagnose raw-FP5D separation for nearby clusters within one y_tar band.

For each preselected pair of clusters that is close in the reconstructed sieve
plane and belongs to the same inferred y_tar band, choose the most separating
pair from the ten untransformed FP5D coordinate pairs.  This is a diagnostic:
the cluster labels are used only to quantify post-clustering separation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABELS = RESULTS / "raw_fullroot_flow_hdbscan_labels.csv"

FEATURES = [
    "P.dc.x_fp",
    "P.dc.y_fp",
    "P.dc.xp_fp",
    "P.dc.yp_fp",
    "P.rb.raster.frybRawAdc",
]
DISPLAY = {
    "P.dc.x_fp": r"$x_{fp}$",
    "P.dc.y_fp": r"$y_{fp}$",
    "P.dc.xp_fp": r"$x'_{fp}$",
    "P.dc.yp_fp": r"$y'_{fp}$",
    "P.rb.raster.frybRawAdc": r"$fr_{ybpm}$",
}

# The four pairs highlighted in same_ytar_band_near_cluster_pairs_on_sieve.png.
PAIRS = [
    ("S1", 219, 220, 0),
    ("S2", 15, 68, 1),
    ("S3", 2, 3, 2),
    ("S4", 45, 56, 2),
]


def fisher_dprime(x: np.ndarray, cls: np.ndarray) -> float:
    xa, xb = x[cls == 0], x[cls == 1]
    cov_a = np.atleast_2d(np.cov(xa, rowvar=False))
    cov_b = np.atleast_2d(np.cov(xb, rowvar=False))
    pooled = 0.5 * (cov_a + cov_b) + np.eye(x.shape[1]) * 1e-10
    delta = xa.mean(axis=0) - xb.mean(axis=0)
    return float(np.sqrt(max(0.0, delta @ np.linalg.pinv(pooled) @ delta)))


def separation_score(x: np.ndarray, cls: np.ndarray) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=25521)
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    probability = cross_val_predict(model, x, cls, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(cls, probability)
    # Cluster identity is arbitrary, so report an orientation-independent AUC.
    return max(float(auc), float(1.0 - auc)), fisher_dprime(x, cls)


def main() -> None:
    df = pd.read_csv(LABELS)
    active = df.loc[df["flow_hdbscan_cluster"] >= 0].copy()
    # Keep the recorded coordinates for the plot.  Scaling is used only while
    # comparing coordinate pairs so their numerical units do not bias ranking.
    raw_features = active[FEATURES].copy()
    scaler = RobustScaler(quantile_range=(5, 95))
    active.loc[:, FEATURES] = scaler.fit_transform(active[FEATURES])

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), constrained_layout=True)
    details: list[dict[str, object]] = []

    for col, (tag, first, second, band) in enumerate(PAIRS):
        part = active.loc[active["flow_hdbscan_cluster"].isin([first, second])].copy()
        cls = (part["flow_hdbscan_cluster"].to_numpy() == second).astype(int)
        sieve = part[["sieve_x", "sieve_y"]].to_numpy()
        centers = np.vstack([sieve[cls == label].mean(axis=0) for label in (0, 1)])
        distance = float(np.linalg.norm(centers[0] - centers[1]))

        rankings = []
        for i, j in itertools.combinations(range(len(FEATURES)), 2):
            x2 = part[[FEATURES[i], FEATURES[j]]].to_numpy()
            auc, dprime = separation_score(x2, cls)
            rankings.append((auc, dprime, i, j))
        auc, dprime, i, j = max(rankings, key=lambda item: (item[0], item[1]))

        top = axes[0, col]
        bottom = axes[1, col]
        for label, cluster in enumerate((first, second)):
            mask = cls == label
            top.scatter(
                sieve[mask, 0], sieve[mask, 1], s=9, alpha=0.78,
                color=cmap(label), label=f"{cluster} (n={mask.sum():,})", linewidths=0,
            )
            top.scatter(
                centers[label, 0], centers[label, 1], marker="*", s=72,
                color="red", edgecolor="white", linewidth=0.45, zorder=5,
            )
        midpoint = centers.mean(axis=0)
        span = max(0.9, distance * 1.9, np.max(np.ptp(sieve, axis=0)) * 0.58)
        top.set_xlim(midpoint[0] - span, midpoint[0] + span)
        top.set_ylim(midpoint[1] - span, midpoint[1] + span)
        top.set_title(
            f"same-band sieve neighbours: {first} vs {second}\n"
            f"centre distance = {distance:.3f} cm (band {band})",
            fontsize=12,
        )
        top.set_xlabel(r"reconstructed $x_{sieve}$")
        if col == 0:
            top.set_ylabel(r"reconstructed $y_{sieve}$")
        top.grid(alpha=0.17)
        top.legend(frameon=False, fontsize=8, loc="upper right")

        x2 = raw_features.loc[part.index, [FEATURES[i], FEATURES[j]]].to_numpy()
        for label, cluster in enumerate((first, second)):
            mask = cls == label
            bottom.scatter(
                x2[mask, 0], x2[mask, 1], s=9, alpha=0.78,
                color=cmap(label), label=str(cluster), linewidths=0,
            )
        bottom.set_title(f"best raw-FP 2D projection\nAUC={auc:.3f}; Fisher d'={dprime:.2f}", fontsize=12)
        bottom.set_xlabel(DISPLAY[FEATURES[i]])
        if col == 0:
            bottom.set_ylabel(DISPLAY[FEATURES[j]])
        bottom.grid(alpha=0.17)

        details.append({
            "pair": tag,
            "inferred_ytar_band": band,
            "clusters": [first, second],
            "events": [int((cls == 0).sum()), int((cls == 1).sum())],
            "sieve_center_distance_cm": distance,
            "best_raw_fp_feature_pair": [FEATURES[i], FEATURES[j]],
            "cross_validated_linear_lda_auc": auc,
            "fisher_dprime": dprime,
        })

    fig.suptitle(
        "Same inferred $y_{tar}$-band sieve neighbours can be separated in selected original FP5D coordinate pairs",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(RESULTS / "same_ytar_band_near_cluster_best_fp2d_separation.png", dpi=200)
    plt.close(fig)
    (RESULTS / "same_ytar_band_near_cluster_best_fp2d_separation.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
