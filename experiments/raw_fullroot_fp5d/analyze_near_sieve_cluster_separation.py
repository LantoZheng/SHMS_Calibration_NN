"""Find near sieve-plane clusters and show their strongest raw-FP 2D separation.

Projection axes are selected only from the ten pairs of the original measured
FP5D coordinates.  No learned embedding or reconstructed quantity is used to
choose the two axes.
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

OUT = Path(__file__).parent / "results"
FEATURES = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
DISPLAY = {"P.dc.x_fp": r"$x_{fp}$", "P.dc.y_fp": r"$y_{fp}$", "P.dc.xp_fp": r"$x'_{fp}$", "P.dc.yp_fp": r"$y'_{fp}$", "P.rb.raster.frybRawAdc": r"$fr_{ybpm}$"}
FOCUS = (219, 220)


def auc_for_pair(x, label):
    cv = StratifiedKFold(5, shuffle=True, random_state=25521)
    scores = cross_val_predict(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), x, label, cv=cv, method="predict_proba")[:, 1]
    return float(max(roc_auc_score(label, scores), 1 - roc_auc_score(label, scores)))


def fisher_distance(x, label):
    a, b = x[label == 0], x[label == 1]
    pooled = .5 * (np.cov(a, rowvar=False) + np.cov(b, rowvar=False))
    pooled += np.eye(2) * 1e-8
    delta = a.mean(axis=0) - b.mean(axis=0)
    return float(np.sqrt(delta @ np.linalg.pinv(pooled) @ delta))


def pair_analysis(df, a, b, scale):
    part = df[df.flow_hdbscan_cluster.isin([a, b])].copy()
    label = (part.flow_hdbscan_cluster.to_numpy() == b).astype(int)
    xs = scale.transform(part[FEATURES])
    scores = []
    for i, j in itertools.combinations(range(len(FEATURES)), 2):
        x2 = xs[:, [i, j]]
        scores.append({"feature_x": FEATURES[i], "feature_y": FEATURES[j], "auc": auc_for_pair(x2, label), "fisher_dprime": fisher_distance(x2, label)})
    rank = pd.DataFrame(scores).sort_values(["auc", "fisher_dprime"], ascending=False).reset_index(drop=True)
    return part, rank


def main():
    df = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_labels.csv")
    active = df[df.flow_hdbscan_cluster >= 0]
    centers = active.groupby("flow_hdbscan_cluster").agg(events=("flow_hdbscan_cluster", "size"), sieve_x=("sieve_x", "median"), sieve_y=("sieve_y", "median"))
    labels = centers.index.to_numpy(); xy = centers[["sieve_x", "sieve_y"]].to_numpy()
    distance = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(axis=2)); np.fill_diagonal(distance, np.inf)
    records = []
    for i, j in zip(*np.where(np.triu(distance < .95, 1))):
        records.append({"cluster_a": int(labels[i]), "cluster_b": int(labels[j]), "sieve_center_distance": float(distance[i,j]), "events_a": int(centers.iloc[i].events), "events_b": int(centers.iloc[j].events)})
    nearby = pd.DataFrame(records).sort_values("sieve_center_distance").reset_index(drop=True)
    if not (((nearby.cluster_a == FOCUS[0]) & (nearby.cluster_b == FOCUS[1])) | ((nearby.cluster_a == FOCUS[1]) & (nearby.cluster_b == FOCUS[0]))).any():
        da = centers.loc[FOCUS[0], ["sieve_x", "sieve_y"]].to_numpy(); db = centers.loc[FOCUS[1], ["sieve_x", "sieve_y"]].to_numpy()
        nearby = pd.concat([nearby, pd.DataFrame([{"cluster_a": FOCUS[0], "cluster_b": FOCUS[1], "sieve_center_distance": float(np.linalg.norm(da-db)), "events_a": int(centers.loc[FOCUS[0], "events"]), "events_b": int(centers.loc[FOCUS[1], "events"])}])], ignore_index=True)
    scale = RobustScaler(quantile_range=(5,95)).fit(active[FEATURES])
    all_ranks = []; selected = []
    for _, row in nearby.head(25).iterrows():
        a, b = int(row.cluster_a), int(row.cluster_b)
        _, rank = pair_analysis(active, a, b, scale)
        best = rank.iloc[0].to_dict(); best.update(row.to_dict()); all_ranks.append(best)
    ranking = pd.DataFrame(all_ranks).sort_values("sieve_center_distance").reset_index(drop=True)
    ranking.to_csv(OUT / "near_sieve_cluster_pair_2d_separation_ranking.csv", index=False)
    # Force the named pair, plus its three closest independent neighbours.
    chosen = [FOCUS]
    for row in nearby.itertuples(index=False):
        p = (int(row.cluster_a), int(row.cluster_b))
        if p != FOCUS and p[::-1] != FOCUS and p not in chosen and p[::-1] not in chosen:
            chosen.append(p)
        if len(chosen) == 4: break

    fig, axes = plt.subplots(2, len(chosen), figsize=(5.25 * len(chosen), 9.0), constrained_layout=True)
    report = []
    for col, (a, b) in enumerate(chosen):
        part, rank = pair_analysis(active, a, b, scale)
        best = rank.iloc[0]; xcol, ycol = best.feature_x, best.feature_y
        ca, cb = centers.loc[a], centers.loc[b]
        d_sieve = float(np.linalg.norm(ca[["sieve_x", "sieve_y"]] - cb[["sieve_x", "sieve_y"]]))
        # Sieve-space context, with a shared local zoom around the two centres.
        ax = axes[0, col]
        ax.scatter(part.loc[part.flow_hdbscan_cluster == a, "sieve_x"], part.loc[part.flow_hdbscan_cluster == a, "sieve_y"], s=4, alpha=.66, color="#2E74B5", label=f"{a} (n={len(part[part.flow_hdbscan_cluster == a])})")
        ax.scatter(part.loc[part.flow_hdbscan_cluster == b, "sieve_x"], part.loc[part.flow_hdbscan_cluster == b, "sieve_y"], s=4, alpha=.66, color="#D55E00", label=f"{b} (n={len(part[part.flow_hdbscan_cluster == b])})")
        ax.scatter([ca.sieve_x, cb.sieve_x], [ca.sieve_y, cb.sieve_y], c="red", s=20, zorder=4)
        midx, midy = (ca.sieve_x + cb.sieve_x) / 2, (ca.sieve_y + cb.sieve_y) / 2
        ax.set(xlim=(midx-1.35, midx+1.35), ylim=(midy-1.35, midy+1.35), xlabel=r"reconstructed $x_{sieve}$", ylabel=r"reconstructed $y_{sieve}$", title=f"sieve neighbours: {a} vs {b}\ncentre distance = {d_sieve:.3f} cm")
        ax.legend(fontsize=7, frameon=False, loc="best"); ax.grid(alpha=.15)
        # Best original-coordinate 2D projection.
        ax = axes[1, col]
        ax.scatter(part.loc[part.flow_hdbscan_cluster == a, xcol], part.loc[part.flow_hdbscan_cluster == a, ycol], s=4, alpha=.55, color="#2E74B5")
        ax.scatter(part.loc[part.flow_hdbscan_cluster == b, xcol], part.loc[part.flow_hdbscan_cluster == b, ycol], s=4, alpha=.55, color="#D55E00")
        ax.set(xlabel=DISPLAY[xcol], ylabel=DISPLAY[ycol], title=f"best raw-FP 2D projection\nAUC={best.auc:.3f}; Fisher d'={best.fisher_dprime:.2f}")
        ax.grid(alpha=.15)
        report.append({"clusters": [a,b], "sieve_center_distance_cm": d_sieve, "best_raw_fp_pair": [xcol,ycol], "cross_validated_2d_lda_auc": float(best.auc), "fisher_dprime": float(best.fisher_dprime), "all_pairs": rank.to_dict(orient="records")})
    fig.suptitle("Clusters close in reconstructed sieve plane can be separated in selected original FP5D coordinate pairs", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "near_sieve_cluster_best_fp2d_separation.png", dpi=250, bbox_inches="tight")
    (OUT / "near_sieve_cluster_best_fp2d_separation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
