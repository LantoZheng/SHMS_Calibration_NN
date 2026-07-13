"""Diffusion/spectral clustering on a self-tuned graph built from raw 5D FP data.

There is no learned output embedding: spectral vectors are an internal graph
partitioning calculation.  Existing sieve labels are read only for post-hoc
evaluation of the resulting partition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FEATURES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
READ = FEATURES + ["foil_position", "cluster"]
SEED = 25521


def load(n=6000):
    root = Path(__file__).resolve().parents[2]
    path = root / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    df = pd.read_csv(path, usecols=READ).dropna().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    # Balance event rates but never expose this group variable to graph construction.
    pieces = [g.iloc[rng.choice(len(g), n // 3, replace=False)] for _, g in df.groupby("foil_position", observed=True)]
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def diffusion_operator(x, neighbors=55, alpha=1.0):
    """Symmetric density-corrected diffusion operator from 5D kNN edges."""
    n = len(x)
    d, ind = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1).fit(x).kneighbors(x)
    rows = np.repeat(np.arange(n), neighbors)
    cols = ind[:, 1:].ravel()
    dist = d[:, 1:].ravel()
    # Zelnik-Manor local scales; the bandwidth has no labels or reconstructed data.
    sigma = d[:, 15]
    weight = np.exp(-(dist**2) / np.maximum(sigma[rows] * sigma[cols], 1e-12))
    w = coo_matrix((weight, (rows, cols)), shape=(n, n)).tocsr()
    w = w.maximum(w.T)
    q = np.asarray(w.sum(axis=1)).ravel()
    qalpha = np.power(np.maximum(q, 1e-12), -alpha)
    w = diags(qalpha) @ w @ diags(qalpha)
    degree = np.asarray(w.sum(axis=1)).ravel()
    dmh = np.power(np.maximum(degree, 1e-12), -0.5)
    return diags(dmh) @ w @ diags(dmh)


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    df = load()
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FEATURES])
    foil = df.foil_position.astype(int).to_numpy()
    reference = foil * 1000 + df.cluster.astype(int).to_numpy()
    op = diffusion_operator(x)
    # Eigenvectors are only transient coordinates used by normalized-cut k-means.
    values, vectors = eigsh(op, k=161, which="LA", tol=1e-4, v0=np.ones(len(x)))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    records = []
    for time in (1, 2, 4):
        for k in (50, 80, 110, 150):
            z = vectors[:, 1:k] * np.power(np.maximum(values[1:k], 0), time)
            z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            labels = KMeans(n_clusters=k, n_init=8, random_state=SEED, algorithm="lloyd").fit_predict(z)
            size = np.bincount(labels)
            records.append({
                "diffusion_time": time, "clusters": k,
                "median_cluster_size": float(np.median(size)), "max_cluster_fraction": float(size.max()/len(labels)),
                "reference_hole_ami": float(adjusted_mutual_info_score(reference, labels)),
                "reference_hole_ari": float(adjusted_rand_score(reference, labels)),
                "reference_foil_ami": float(adjusted_mutual_info_score(foil, labels)),
            })
    scan = pd.DataFrame(records)
    scan.to_csv(out / "diffusion_spectral_geometry_scan.csv", index=False)
    best = scan.sort_values(["reference_hole_ami", "reference_hole_ari"], ascending=False).head(1)
    gaps = values[:-1] - values[1:]
    candidates = np.argsort(gaps[1:150])[::-1][:8] + 2
    report = {
        "n_events": int(len(df)), "features": FEATURES,
        "graph": "55-NN, self-tuned local Gaussian affinity, alpha=1 density normalization in raw robust-scaled 5D",
        "clustering": "diffusion-operator eigensystem used internally for normalized-cut k-means; no persisted low-dimensional feature space",
        "evaluation_only": "existing sieve cluster / foil labels",
        "best_scan": best.to_dict(orient="records")[0],
        "largest_eigengap_candidate_counts": [int(x) for x in candidates],
        "top_eigenvalues": [float(x) for x in values[:12]],
        "verdict": "same-run diagnostic; compare only against the existing 5D graph baselines before treating as a clustering candidate."
    }
    (out / "diffusion_spectral_geometry_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
