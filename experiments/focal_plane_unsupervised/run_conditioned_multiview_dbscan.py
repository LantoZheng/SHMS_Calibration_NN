#!/usr/bin/env python3
"""Conditioned FP-kNN clustering with sieve as a weak edge prior.

The three coarse groups are inferred from reconstructed P_gtr_y by a 1-D GMM.
Existing sieve-HDBSCAN cluster identifiers are loaded only for report-time
comparison; neither they nor foil_position enter the graph or DBSCAN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
SIEVE = ["sieve_x", "sieve_y"]
READ = FP + SIEVE + ["P_gtr_y", "foil_position", "cluster"]


def graph(fp: np.ndarray, sieve: np.ndarray, weight: float, neighbors: int = 60):
    fp = RobustScaler(quantile_range=(5, 95)).fit_transform(fp)
    sieve = RobustScaler(quantile_range=(5, 95)).fit_transform(sieve)
    dist, ind = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1).fit(fp).kneighbors(fp)
    rows = np.repeat(np.arange(len(fp)), neighbors)
    cols = ind[:, 1:].ravel()
    edge = np.sqrt(dist[:, 1:].ravel() ** 2 + weight * np.sum((sieve[rows] - sieve[cols]) ** 2, axis=1))
    directed = coo_matrix((edge, (rows, cols)), shape=(len(fp), len(fp))).tocsr()
    joined = directed.maximum(directed.T)
    joined.setdiag(0.0)
    joined.sort_indices()
    return fp, joined, edge


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(path, usecols=READ).dropna()
    rng = np.random.default_rng(25521)
    frame = frame.iloc[rng.choice(len(frame), 18000, replace=False)].reset_index(drop=True)

    # Coarse conditioning is reconstructed-only; no existing foil classification is used here.
    gmm = GaussianMixture(n_components=3, random_state=25521).fit(frame[["P_gtr_y"]])
    groups = gmm.predict(frame[["P_gtr_y"]])
    reference = (frame.foil_position.astype(int) * 1000 + frame.cluster.astype(int)).to_numpy()
    cache = {}
    for group in range(3):
        mask = groups == group
        cache[group] = graph(frame.loc[mask, FP].to_numpy(), frame.loc[mask, SIEVE].to_numpy(), 0.0)

    rows = []
    for weight in (0.0, 0.01, 0.02, 0.05, 0.10):
        graphs = {}
        scales = []
        for group in range(3):
            mask = groups == group
            fp, sparse, edges = graph(frame.loc[mask, FP].to_numpy(), frame.loc[mask, SIEVE].to_numpy(), weight)
            graphs[group] = (mask, fp, sparse)
            scales.append(np.quantile(edges, 0.78))
        base_eps = float(np.median(scales))
        for eps_factor in (0.75, 0.90, 1.05):
            for min_samples in (8, 16, 32):
                output = np.full(len(frame), -1, dtype=int)
                next_label = 0
                for group, (mask, _, sparse) in graphs.items():
                    labels = DBSCAN(eps=base_eps * eps_factor, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(sparse)
                    active = labels >= 0
                    n_local = int(labels[active].max() + 1) if active.any() else 0
                    labels[active] += next_label
                    next_label += n_local
                    output[mask] = labels
                active = output >= 0
                sizes = np.bincount(output[active]) if active.any() else np.array([])
                rows.append({
                    "prior_weight": weight, "eps": base_eps * eps_factor, "min_samples": min_samples,
                    "clusters": int(len(sizes)), "noise_fraction": float(1 - active.mean()),
                    "max_cluster_fraction": float(sizes.max() / len(output)) if len(sizes) else 0.0,
                    "reference_ami": float(adjusted_mutual_info_score(reference, output)),
                    "reference_ari": float(adjusted_rand_score(reference, output)),
                })
    scan = pd.DataFrame(rows)
    scan.to_csv(out / "conditioned_multiview_scan.csv", index=False)
    report = {
        "n_events": int(len(frame)),
        "features_for_clustering": FP,
        "weak_prior": SIEVE,
        "coarse_conditioning": "three-component GMM on reconstructed P_gtr_y",
        "not_used_to_construct_clusters": ["foil_position", "cluster", "hole_id", "hole_row", "hole_col"],
        "reference_evaluation_only": "existing sieve-plane HDBSCAN cluster identity",
        "gmm_means_p_gtr_y": sorted(float(x) for x in gmm.means_.ravel()),
        "best_reference_ami_diagnostic": scan.sort_values("reference_ami", ascending=False).head(1).to_dict(orient="records")[0],
        "best_reference_ari_diagnostic": scan.sort_values("reference_ari", ascending=False).head(1).to_dict(orient="records")[0],
    }
    (out / "conditioned_multiview_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
