"""Build a local FP5D branch coordinate for close sieve-neighbour cluster pairs.

For each local sieve neighbourhood, the provisional HDBSCAN branches provide
weak labels.  The Fisher/LDA normal coordinate is then a one-dimensional local
chart used only to decide whether the two provisional branches should remain
separate.  It is intentionally not advertised as a single global z coordinate.
"""

from __future__ import annotations

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
FEATURES = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
PAIRS = [("S1", 219, 220, 0), ("S2", 15, 68, 1), ("S3", 2, 3, 2), ("S4", 45, 56, 2)]


def auc_abs(score: np.ndarray, target: np.ndarray) -> float:
    value = roc_auc_score(target, score)
    return float(max(value, 1 - value))


def dprime(value: np.ndarray, target: np.ndarray) -> float:
    a, b = value[target == 0], value[target == 1]
    return float(abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)) + 1e-12))


def main() -> None:
    df = pd.read_csv(RESULTS / "raw_fullroot_flow_hdbscan_labels.csv")
    data = df.loc[df.flow_hdbscan_cluster >= 0].copy()
    scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(data[FEATURES])
    scaled = pd.DataFrame(scaled, index=data.index, columns=FEATURES)
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(21, 12), constrained_layout=True)
    colours = ["tab:blue", "tab:orange"]
    for col, (tag, first, second, band) in enumerate(PAIRS):
        part = data.loc[data.flow_hdbscan_cluster.isin([first, second])]
        target = (part.flow_hdbscan_cluster.to_numpy() == second).astype(int)
        x = scaled.loc[part.index].to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=25521)
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        # OOF score is used for an honest diagnostic, rather than in-sample fit.
        score = cross_val_predict(model, x, target, cv=cv, method="decision_function")
        # Choose a consistent sign: higher score for the cluster with larger mean y_tar.
        if part.loc[target == 1, "P.gtr.y"].mean() < part.loc[target == 0, "P.gtr.y"].mean():
            score *= -1
        data.loc[part.index, f"local_z3_{tag}"] = score
        center = part.groupby("flow_hdbscan_cluster")[["sieve_x", "sieve_y"]].mean()
        distance = float(np.linalg.norm(center.loc[first] - center.loc[second]))

        for value, cluster, colour in zip((0, 1), (first, second), colours):
            mask = target == value
            axes[0, col].scatter(part.loc[mask, "sieve_x"], part.loc[mask, "sieve_y"], s=10, alpha=.75, color=colour, label=str(cluster), linewidths=0)
            axes[0, col].scatter(center.loc[cluster, "sieve_x"], center.loc[cluster, "sieve_y"], marker="*", s=80, color="red", edgecolor="white", linewidth=.4)
        mid, span = center.mean(axis=0), max(.9, distance * 1.9, np.ptp(part[["sieve_x", "sieve_y"]], axis=0).max()*.58)
        axes[0, col].set(xlim=(mid.sieve_x-span, mid.sieve_x+span), ylim=(mid.sieve_y-span, mid.sieve_y+span), xlabel=r"reconstructed $x_{sieve}$", title=f"{tag}: {first} vs {second}, band {band}\ncentre distance={distance:.3f} cm")
        axes[0, col].grid(alpha=.16); axes[0, col].legend(frameon=False, fontsize=8)
        if col == 0: axes[0, col].set_ylabel(r"reconstructed $y_{sieve}$")

        for row, values, name, label in [
            (1, part["P.gtr.y"].to_numpy(), "reconstructed_ytar", r"reconstructed $y_{tar}$"),
            (2, score, "local_discriminant_z3", r"local FP5D $z_3^{(local)}$"),
        ]:
            metrics = {"auc": auc_abs(values, target), "dprime": dprime(values, target)}
            for value, cluster, colour in zip((0, 1), (first, second), colours):
                vals = values[target == value]
                bins = np.linspace(values.min(), values.max(), 30)
                axes[row, col].hist(vals, bins=bins, histtype="stepfilled", alpha=.35, color=colour)
            axes[row, col].set(title=f"{label}: AUC={metrics['auc']:.3f}; d'={metrics['dprime']:.2f}", xlabel=label)
            axes[row, col].grid(alpha=.16)
            if col == 0: axes[row, col].set_ylabel("events / bin")
            rows.append({"pair": tag, "clusters": f"{first} vs {second}", "coordinate": name, **metrics})

    fig.suptitle("A local FP5D Fisher-normal coordinate resolves close sieve branches when reconstructed $y_{tar}$ is ambiguous", fontsize=17, fontweight="bold")
    fig.savefig(RESULTS / "local_discriminant_z3_same_band_near_pairs.png", dpi=200)
    pd.DataFrame(rows).to_csv(RESULTS / "local_discriminant_z3_pair_metrics.csv", index=False)
    (RESULTS / "local_discriminant_z3_method.json").write_text(json.dumps({
        "coordinate": "z3_local = w_local^T robust_scaled(FP5D) + b_local",
        "w_local": "shrinkage Fisher/LDA normal fitted within each local sieve neighbourhood from provisional HDBSCAN branch labels",
        "evaluation": "five-fold out-of-fold decision score; sign oriented by local reconstructed ytar mean",
        "use": "local cluster refinement; not a globally meaningful coordinate",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
