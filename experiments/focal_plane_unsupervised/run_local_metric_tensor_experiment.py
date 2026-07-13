#!/usr/bin/env python3
"""Piecewise local 5D metric tensors; no dimensionality reduction is used."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
ADC = ["P_rb_raster_fryaRawAdc", "P_rb_raster_frybRawAdc"]
TARGET = ["cluster_center_x", "cluster_center_y", "foil_ytar_center"]


def fit_local_l(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d, L = x.shape[1], np.eye(x.shape[1])
    m = np.zeros_like(L); v = np.zeros_like(L)
    for t in range(1, 501):
        a = rng.integers(0, len(x), size=4096); b = rng.integers(0, len(x), size=4096)
        dx, dy = x[a] - x[b], y[a] - y[b]
        M = L @ L.T
        err = np.einsum("bi,ij,bj->b", dx, M, dx) - np.sum(dy * dy, axis=1)
        gm = 2 * np.einsum("b,bi,bj->ij", err, dx, dx) / len(a) + .05 * (M - np.eye(d))
        g = 2 * gm @ L
        m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
        L -= .012 * (m / (1 - .9**t)) / (np.sqrt(v / (1 - .999**t)) + 1e-8)
    return L


def local_neighbors(x: np.ndarray, patch: np.ndarray, metrics: list[np.ndarray], candidates: int = 120) -> np.ndarray:
    raw = NearestNeighbors(n_neighbors=candidates + 1, n_jobs=-1).fit(x).kneighbors(return_distance=False)[:, 1:]
    output = np.empty((len(x), 20), dtype=int)
    matrices = [l @ l.T for l in metrics]
    for i, pool in enumerate(raw):
        dx = x[i] - x[pool]
        m = 0.5 * (matrices[patch[i]][None, :, :] + np.array([matrices[p] for p in patch[pool]]))
        dist = np.einsum("bi,bij,bj->b", dx, m, dx)
        output[i] = pool[np.argsort(dist)[:20]]
    return output


def score(neighbors: np.ndarray, target: np.ndarray, labels: np.ndarray, x: np.ndarray) -> dict[str, float]:
    target_n = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(target).kneighbors(return_distance=False)[:, 1:]
    overlap = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(neighbors, target_n)])
    # silhouette uses an approximate local chart transform only for reporting; neighbor metrics are primary.
    return {"target_knn_overlap_k20": float(overlap), "same_cluster_fraction_k20": float((labels[:, None] == labels[neighbors]).mean())}


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
    held = rng.choice(np.unique(labels), size=44, replace=False)
    test = np.isin(labels, held); train = ~test
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP + ["fry_proxy"]])
    y = StandardScaler().fit_transform(df[TARGET])
    kmeans = KMeans(n_clusters=12, n_init=10, random_state=25521).fit(x[train])
    train_patch = kmeans.labels_; test_patch = kmeans.predict(x[test])
    metrics = []
    for patch in range(12):
        use = train_patch == patch
        metrics.append(fit_local_l(x[train][use], y[train][use], seed=25521 + patch))
    raw_neighbors = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(x[test]).kneighbors(return_distance=False)[:, 1:]
    tensor_neighbors = local_neighbors(x[test], test_patch, metrics)
    report = {
        "dimension": 5, "method": "12 piecewise SPD 5x5 metric tensors, candidate graph from raw 120-NN",
        "input_features": FP + ["fry_proxy"], "weak_target_distance": TARGET,
        "split": "44 entire sieve clusters held out", "raw": score(raw_neighbors, y[test], labels[test], x[test]),
        "local_metric_tensor": score(tensor_neighbors, y[test], labels[test], x[test]),
        "patch_train_sizes": [int((train_patch == p).sum()) for p in range(12)],
    }
    (out / "local_metric_tensor_holeholdout_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
