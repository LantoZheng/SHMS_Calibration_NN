#!/usr/bin/env python3
"""Learn a full-rank 5D Mahalanobis metric from weak sieve+ytar distances.

The representation remains 5D: z = x L.  L is constrained by a penalty
toward the identity, so this experiment reweights/shears the original measured
coordinates rather than projecting them to a lower-dimensional embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
ADC = ["P_rb_raster_fryaRawAdc", "P_rb_raster_frybRawAdc"]
TARGET = ["cluster_center_x", "cluster_center_y", "foil_ytar_center"]


def evaluate(z: np.ndarray, y: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    nn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(z).kneighbors(return_distance=False)[:, 1:]
    ynn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(y).kneighbors(return_distance=False)[:, 1:]
    overlap = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(nn, ynn)])
    sample = np.linspace(0, len(z) - 1, min(5000, len(z)), dtype=int)
    return {
        "target_knn_overlap_k20": float(overlap),
        "same_cluster_fraction_k20": float((labels[:, None] == labels[nn]).mean()),
        "silhouette": float(silhouette_score(z[sample], labels[sample])),
    }


def train_metric(x: np.ndarray, y: np.ndarray, steps: int = 1000, batch: int = 8192, seed: int = 25521) -> np.ndarray:
    """Adam optimization of L, where d_M^2 = ||(xi-xj)L||^2."""
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    L = np.eye(d)
    m, v = np.zeros_like(L), np.zeros_like(L)
    lr, beta1, beta2, eps, reg = 0.015, .9, .999, 1e-8, .02
    for step in range(1, steps + 1):
        # Mix local and random pairs; local pairs prevent the global spread from dominating.
        left = rng.integers(0, len(x), size=batch)
        right = rng.integers(0, len(x), size=batch)
        dx = x[left] - x[right]
        dy = y[left] - y[right]
        target_d2 = np.sum(dy * dy, axis=1)
        M = L @ L.T
        predicted_d2 = np.einsum("bi,ij,bj->b", dx, M, dx)
        error = predicted_d2 - target_d2
        grad_m = 2.0 * np.einsum("b,bi,bj->ij", error, dx, dx) / batch + 2 * reg * (M - np.eye(d))
        grad_l = 2 * grad_m @ L
        m = beta1 * m + (1 - beta1) * grad_l
        v = beta2 * v + (1 - beta2) * (grad_l * grad_l)
        mh = m / (1 - beta1**step)
        vh = v / (1 - beta2**step)
        L -= lr * mh / (np.sqrt(vh) + eps)
    return L


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    read = FP + ADC + TARGET + ["foil_position", "cluster"]
    df = pd.read_csv(source, usecols=read).dropna()
    rng = np.random.default_rng(25521)
    df = df.iloc[rng.choice(len(df), 60000, replace=False)].reset_index(drop=True)
    df["fry_proxy"] = df[ADC].mean(axis=1)
    labels = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    holes = np.unique(labels)
    held = rng.choice(holes, size=int(np.ceil(.20 * len(holes))), replace=False)
    test = np.isin(labels, held)
    train = ~test
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP + ["fry_proxy"]])
    y = StandardScaler().fit_transform(df[TARGET])
    L = train_metric(x[train], y[train])
    z = x @ L
    report = {
        "dimension": 5,
        "map": "z = x L, full-rank 5x5 matrix",
        "input_features": FP + ["fry_proxy"], "weak_target_distance": TARGET,
        "split": "entire held-out sieve clusters excluded from metric fitting",
        "held_out_sieve_clusters": int(len(held)),
        "raw": evaluate(x[test], y[test], labels[test]),
        "fullrank_metric": evaluate(z[test], y[test], labels[test]),
        "metric_matrix": (L @ L.T).tolist(),
        "transform_condition_number": float(np.linalg.cond(L)),
    }
    (out / "fullrank_metric_holeholdout_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
