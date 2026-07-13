#!/usr/bin/env python3
"""Focal-plane clustering with sieve coordinates used only as a soft prior.

No hole IDs, foil IDs, mechanical-grid values, or target labels are read.
Candidate graph edges are formed exclusively from focal-plane kNN; the current
sieve reconstruction only increases the distance along those existing edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
SIEVE = ["sieve_x", "sieve_y"]


def load(max_events: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    # Explicitly limited to measured FP variables plus the weak reconstructed sieve prior.
    frame = pd.read_csv(path, usecols=FP + SIEVE).dropna()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(frame), size=min(max_events, len(frame)), replace=False)
    data = frame.iloc[chosen]
    return data[FP].to_numpy(np.float64), data[SIEVE].to_numpy(np.float64)


def joint_knn_distance(fp: np.ndarray, sieve: np.ndarray, prior_weight: float, neighbors: int = 40):
    """Sparse distance graph: no sieve-only edge can ever be created."""
    fp_scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(fp)
    sieve_scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(sieve)
    nn = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1).fit(fp_scaled)
    fp_d, ind = nn.kneighbors(fp_scaled)
    rows = np.repeat(np.arange(len(fp_scaled)), neighbors)
    cols = ind[:, 1:].ravel()
    d_fp = fp_d[:, 1:].ravel()
    d_sieve = np.linalg.norm(sieve_scaled[rows] - sieve_scaled[cols], axis=1)
    distance = np.sqrt(d_fp**2 + prior_weight * d_sieve**2)
    # Symmetric union, not intersection: every retained edge originated in FP kNN.
    directed = coo_matrix((distance, (rows, cols)), shape=(len(fp_scaled), len(fp_scaled))).tocsr()
    graph = directed.maximum(directed.T)
    graph.setdiag(0.0)
    graph.sort_indices()
    return fp_scaled, graph.tocsr(), distance


def stats(fp: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    keep = labels >= 0
    sizes = np.bincount(labels[keep]) if keep.any() else np.array([])
    clusters = len(sizes)
    sil = None
    if clusters >= 2 and keep.sum() >= 100:
        sample = np.linspace(0, keep.sum() - 1, min(4000, keep.sum()), dtype=int)
        sil = float(silhouette_score(fp[keep][sample], labels[keep][sample]))
    return {
        "clusters": int(clusters), "noise_fraction": float(1 - keep.mean()),
        "max_cluster_fraction": float(sizes.max() / len(fp)) if len(sizes) else 0.0,
        "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "silhouette_in_fp": sil,
    }


def stability(fp: np.ndarray, sieve: np.ndarray, weight: float, eps: float, min_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(fp)
    a = np.sort(rng.choice(n, int(0.8 * n), replace=False))
    b = np.sort(rng.choice(n, int(0.8 * n), replace=False))
    _, ga, _ = joint_knn_distance(fp[a], sieve[a], weight)
    _, gb, _ = joint_knn_distance(fp[b], sieve[b], weight)
    la = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(ga)
    lb = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(gb)
    _, ia, ib = np.intersect1d(a, b, return_indices=True)
    return float(adjusted_rand_score(la[ia], lb[ib]))


def main() -> None:
    fp, sieve = load(max_events=12000, seed=25521)
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for weight in (0.0, 0.02, 0.05, 0.10, 0.20):
        fp_scaled, graph, directed_dist = joint_knn_distance(fp, sieve, weight)
        eps_grid = np.unique(np.quantile(directed_dist, [0.50, 0.65, 0.78, 0.88, 0.94]))
        for min_samples in (10, 20, 40):
            for eps in eps_grid:
                labels = DBSCAN(eps=float(eps), min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(graph)
                row = {"prior_weight": weight, "eps": float(eps), "min_samples": min_samples}
                row.update(stats(fp_scaled, labels))
                row["overlap_ari"] = stability(fp, sieve, weight, float(eps), min_samples, 5000 + min_samples)
                rows.append(row)
    scan = pd.DataFrame(rows)
    scan.to_csv(out / "joint_prior_parameter_scan.csv", index=False)
    eligible = scan[
        (scan.clusters >= 2) & (scan.noise_fraction < 0.50)
        & (scan.max_cluster_fraction < 0.50) & (scan.silhouette_in_fp > 0.10)
    ]
    chosen = eligible.sort_values(["overlap_ari", "silhouette_in_fp"], ascending=False).head(1)
    summary = {
        "focal_features": FP,
        "weak_prior_features": SIEVE,
        "not_read": ["hole_id", "hole_row", "hole_col", "foil_position", "P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_gtr_y"],
        "n_events": len(fp),
        "method": "FP-kNN sparse graph; sieve can only penalize existing FP-neighbor edges",
        "selection_rule": "noise<0.50, no cluster>50%, FP silhouette>0.10, then maximize bootstrap overlap ARI",
        "chosen_candidate": chosen.to_dict(orient="records")[0] if not chosen.empty else None,
        "verdict": "usable weak-prior clustering candidate found" if not chosen.empty else "no usable weak-prior candidate in this scan",
    }
    (out / "joint_prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
