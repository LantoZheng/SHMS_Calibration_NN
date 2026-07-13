#!/usr/bin/env python3
"""Label-free focal-plane density-clustering experiment for run 25521."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


FEATURES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    return argparse.ArgumentParser(description=__doc__).parse_args()


def load_features(repo: Path, max_events: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = repo / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    # usecols is intentional: no sieve, foil, hole, or target-reconstruction fields enter this experiment.
    frame = pd.read_csv(data, usecols=FEATURES).dropna().reset_index(names="event_index")
    rng = np.random.default_rng(seed)
    take = min(max_events, len(frame))
    chosen = rng.choice(len(frame), size=take, replace=False)
    sampled = frame.iloc[np.sort(chosen)]
    return sampled["event_index"].to_numpy(), sampled[FEATURES].to_numpy(dtype=np.float64)


def cluster_stats(x: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    non_noise = labels >= 0
    count = len(set(labels[non_noise]))
    sizes = np.bincount(labels[non_noise]) if non_noise.any() else np.array([])
    silhouette: float | None = None
    if count >= 2 and non_noise.sum() >= 100:
        idx = np.linspace(0, non_noise.sum() - 1, min(5000, non_noise.sum()), dtype=int)
        points = x[non_noise][idx]
        labs = labels[non_noise][idx]
        if len(set(labs)) >= 2:
            silhouette = float(silhouette_score(points, labs))
    return {
        "clusters": int(count),
        "noise_fraction": float(1.0 - non_noise.mean()),
        "min_cluster_size": int(sizes.min()) if len(sizes) else 0,
        "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "silhouette_non_noise": silhouette,
    }


def overlap_stability(x: np.ndarray, eps: float, min_samples: int, seed: int) -> float | None:
    rng = np.random.default_rng(seed)
    n = len(x)
    a = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
    b = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
    la = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(x[a])
    lb = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(x[b])
    common, ia, ib = np.intersect1d(a, b, return_indices=True)
    if len(common) < 100:
        return None
    return float(adjusted_rand_score(la[ia], lb[ib]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=25521)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)

    event_index, raw = load_features(repo, args.max_events, args.seed)
    scaler = RobustScaler(quantile_range=(5, 95))
    x = scaler.fit_transform(raw)
    pca = PCA(n_components=4, random_state=args.seed).fit(x)
    knn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(x)
    dists, _ = knn.kneighbors(x)
    kdist = dists[:, -1]
    eps_values = np.unique(np.quantile(kdist, [0.50, 0.60, 0.70, 0.80, 0.88, 0.93, 0.97]))

    rows: list[dict[str, float | int | None]] = []
    for min_samples in (10, 20, 40):
        for eps in eps_values:
            labels = DBSCAN(eps=float(eps), min_samples=min_samples, n_jobs=-1).fit_predict(x)
            row: dict[str, float | int | None] = {"eps": float(eps), "min_samples": min_samples}
            row.update(cluster_stats(x, labels))
            row["overlap_ari"] = overlap_stability(x, float(eps), min_samples, args.seed + min_samples)
            rows.append(row)

    report = pd.DataFrame(rows)
    # A stable single connected acceptance region is not a useful discrete clustering result.
    eligible = report[
        (report.clusters >= 2)
        & (report.noise_fraction < 0.50)
        & (report.max_cluster_size < 0.50 * len(x))
        & (report.silhouette_non_noise > 0.10)
    ]
    chosen = eligible.sort_values(["overlap_ari", "silhouette_non_noise"], ascending=False).head(1)
    report.to_csv(out / "parameter_scan.csv", index=False)
    summary = {
        "feature_columns": FEATURES,
        "forbidden_columns_not_read": ["sieve_x", "sieve_y", "hole_id", "foil_position", "P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_gtr_y"],
        "n_events": int(len(x)),
        "robust_scaler_quantiles": [5, 95],
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "k20_distance_quantiles": {str(q): float(np.quantile(kdist, q)) for q in (0.5, 0.8, 0.95)},
        "selection_rule": "require no giant cluster, noise<0.50, silhouette>0.10; then maximize overlap ARI. No desired cluster count is used.",
        "chosen_candidate": chosen.to_dict(orient="records")[0] if not chosen.empty else None,
        "verdict": (
            "stable, non-degenerate discrete focal-plane clusters found"
            if not chosen.empty
            else "no parameter set met the label-free discrete-cluster criteria"
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
