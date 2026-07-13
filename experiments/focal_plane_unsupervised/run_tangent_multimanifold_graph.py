"""Tangent-aware multi-manifold graph clustering in the original measured 5D space.

Graph construction deliberately reads only focal-plane coordinates.  Existing
sieve clusters and foil positions are loaded only after clustering to measure
how the unsupervised partition relates to the established reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FEATURES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
READ = FEATURES + ["sieve_x", "sieve_y", "foil_position", "cluster"]
SEED = 25521


def load(n=9000):
    root = Path(__file__).resolve().parents[2]
    data = root / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    df = pd.read_csv(data, usecols=READ).dropna().reset_index(drop=True)
    # The equal-sized foil draw is solely to prevent event rate from determining
    # graph degree.  Its identity is never given to the graph algorithm.
    rng = np.random.default_rng(SEED)
    blocks = []
    for _, g in df.groupby("foil_position", observed=True):
        blocks.append(g.iloc[rng.choice(len(g), min(n // 3, len(g)), replace=False)])
    return pd.concat(blocks, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_geometry(x, neighbors=45, tangent_dim=4):
    """Return fixed FP kNN edges, local Mahalanobis distances and tangent gaps."""
    nn = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1).fit(x)
    euclid, ind = nn.kneighbors(x)
    n, d = x.shape
    local = ind[:, 1:]
    cov_inv = np.empty((n, d, d))
    projectors = np.empty((n, d, d))
    for i in range(n):
        z = x[np.r_[i, local[i]]]
        c = np.cov(z, rowvar=False)
        # Trace-scaled ridge makes the metric stable in a thin local tube.
        c += np.eye(d) * max(np.trace(c) / d, 1e-8) * 0.04
        cov_inv[i] = np.linalg.inv(c)
        _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
        u = vt[:tangent_dim].T
        projectors[i] = u @ u.T
    rows = np.repeat(np.arange(n), neighbors)
    cols = local.ravel()
    delta = x[rows] - x[cols]
    q1 = np.einsum("ni,nij,nj->n", delta, cov_inv[rows], delta)
    q2 = np.einsum("ni,nij,nj->n", delta, cov_inv[cols], delta)
    mahal = np.sqrt(np.maximum((q1 + q2) / 2, 1e-12))
    pdiff = projectors[rows] - projectors[cols]
    tangent_gap = np.sqrt(np.einsum("nij,nij->n", pdiff, pdiff) / (2 * tangent_dim))
    # Self-tuned scale remains in the original 5D graph; it is not an embedding.
    sigma = np.median(mahal.reshape(n, neighbors), axis=1)
    return rows, cols, mahal, tangent_gap, sigma


def graph_labels(n, rows, cols, mahal, tangent_gap, sigma, tangent_weight, resolution):
    scale = sigma[rows] * sigma[cols]
    affinity = np.exp(-(mahal**2) / np.maximum(scale, 1e-12))
    affinity *= np.exp(-tangent_weight * tangent_gap**2)
    # Union of directed kNN edges; repeated edges use their strongest affinity.
    edge = {}
    for a, b, w in zip(rows, cols, affinity):
        key = (int(a), int(b)) if a < b else (int(b), int(a))
        edge[key] = max(edge.get(key, 0.0), float(w))
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_weighted_edges_from((a, b, w) for (a, b), w in edge.items())
    communities = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=SEED)
    labels = np.empty(n, dtype=int)
    for c, members in enumerate(communities):
        labels[list(members)] = c
    return labels, g.number_of_edges()


def scores(labels, reference, foil):
    counts = np.bincount(labels)
    return {
        "clusters": int(len(counts)), "median_cluster_size": float(np.median(counts)),
        "max_cluster_fraction": float(counts.max() / len(labels)),
        "reference_hole_ami": float(adjusted_mutual_info_score(reference, labels)),
        "reference_hole_ari": float(adjusted_rand_score(reference, labels)),
        "reference_foil_ami": float(adjusted_mutual_info_score(foil, labels)),
    }


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    df = load()
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FEATURES])
    foil = df.foil_position.astype(int).to_numpy()
    reference = (foil * 1000 + df.cluster.astype(int).to_numpy())
    rows, cols, mahal, tangent_gap, sigma = build_geometry(x)
    result = []
    for tw in (0.0, 2.0, 6.0):
        for resolution in (1.0, 2.0, 4.0, 8.0, 12.0):
            labels, edges = graph_labels(len(x), rows, cols, mahal, tangent_gap, sigma, tw, resolution)
            row = {"tangent_weight": tw, "resolution": resolution, "edges": edges}
            row.update(scores(labels, reference, foil))
            result.append(row)
    scan = pd.DataFrame(result)
    scan.to_csv(out / "tangent_multimanifold_graph_scan.csv", index=False)
    # Reference score is diagnostic only; candidate selection insists on a
    # non-degenerate, hole-like scale, then prefers AMI over label identity.
    eligible = scan[(scan.clusters >= 80) & (scan.clusters <= 500) & (scan.max_cluster_fraction < .08)]
    best = eligible.sort_values(["reference_hole_ami", "reference_hole_ari"], ascending=False).head(1)
    baseline = scan[scan.tangent_weight == 0].sort_values("reference_hole_ami", ascending=False).head(1)
    report = {
        "n_events": int(len(x)), "features": FEATURES,
        "graph": "45-NN in robust-scaled raw 5D; symmetric local Mahalanobis affinity with self-tuned local scale",
        "tangent": "4D local PCA tangent projector; mismatch downweights, never creates, FP-neighbor edges",
        "clustering": "weighted Louvain graph partition; no foil/hole/sieve/reconstructed-target features",
        "evaluation_only": "existing sieve-plane cluster and foil labels",
        "best_tangent_aware": best.to_dict(orient="records")[0] if not best.empty else None,
        "best_distance_only_baseline": baseline.to_dict(orient="records")[0],
        "verdict": "compare best_tangent_aware against the tangent_weight=0 graph baseline; this is an exploratory same-run diagnostic, not external validation."
    }
    (out / "tangent_multimanifold_graph_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
