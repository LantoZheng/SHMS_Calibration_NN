"""Local sparse subspace clustering (SSC) on raw 5D focal-plane measurements.

For every event, a 50-NN raw-5D candidate set is built.  Sparse *affine*
self-expression selects only a few of those edges. Labels are loaded only to
evaluate the graph partition after it has been made.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FEATURES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
READ = FEATURES + ["foil_position", "cluster"]
SEED = 25521


def load(n=9000):
    root = Path(__file__).resolve().parents[2]
    path = root / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    df = pd.read_csv(path, usecols=READ).dropna().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    chunks = [g.iloc[rng.choice(len(g), n // 3, replace=False)] for _, g in df.groupby("foil_position", observed=True)]
    return pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def sparse_edges(x, alpha, neighbors=50, affine_strength=200.0):
    """L1 affine self-expression restricted to raw-5D kNN candidate points."""
    n = len(x)
    _, ind = NearestNeighbors(n_neighbors=neighbors + 1, n_jobs=-1).fit(x).kneighbors(x)
    rows, cols, values, supports = [], [], [], []
    # The appended row implements sum(coefficients) ~= 1, making the local
    # representation translation-invariant rather than tied to the FP origin.
    add_row = np.sqrt(affine_strength) * np.ones((1, neighbors))
    target_tail = np.array([np.sqrt(affine_strength)])
    for i in range(n):
        candidate = ind[i, 1:]
        design = np.vstack([x[candidate].T, add_row])
        target = np.r_[x[i], target_tail]
        # Features are candidate coefficients; fit_intercept must remain off.
        coef = Lasso(alpha=alpha, fit_intercept=False, max_iter=1500, tol=1e-4).fit(design, target).coef_
        use = np.flatnonzero(np.abs(coef) > 1e-7)
        rows.extend([i] * len(use)); cols.extend(candidate[use]); values.extend(np.abs(coef[use]))
        supports.append(len(use))
    return np.asarray(rows), np.asarray(cols), np.asarray(values), np.asarray(supports)


def partition(n, rows, cols, values, resolution):
    edge = {}
    for a, b, w in zip(rows, cols, values):
        key = (int(a), int(b)) if a < b else (int(b), int(a))
        edge[key] = edge.get(key, 0.0) + float(w)
    g = nx.Graph(); g.add_nodes_from(range(n)); g.add_weighted_edges_from((a, b, w) for (a, b), w in edge.items())
    comm = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=SEED)
    labels = np.empty(n, dtype=int)
    for c, members in enumerate(comm): labels[list(members)] = c
    return labels, g.number_of_edges()


def main():
    out = Path(__file__).parent / "results"; out.mkdir(exist_ok=True)
    df = load()
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FEATURES])
    foil = df.foil_position.astype(int).to_numpy()
    reference = foil * 1000 + df.cluster.astype(int).to_numpy()
    records = []
    for alpha in (0.0005, 0.002, 0.008):
        rows, cols, vals, support = sparse_edges(x, alpha)
        for res in (2.0, 4.0, 8.0, 12.0):
            labels, edges = partition(len(x), rows, cols, vals, res)
            sizes = np.bincount(labels)
            records.append({
                "lasso_alpha": alpha, "resolution": res, "edges": edges,
                "mean_sparse_support": float(support.mean()), "median_sparse_support": float(np.median(support)),
                "clusters": int(len(sizes)), "median_cluster_size": float(np.median(sizes)),
                "max_cluster_fraction": float(sizes.max()/len(labels)),
                "reference_hole_ami": float(adjusted_mutual_info_score(reference, labels)),
                "reference_hole_ari": float(adjusted_rand_score(reference, labels)),
                "reference_foil_ami": float(adjusted_mutual_info_score(foil, labels)),
            })
    scan = pd.DataFrame(records); scan.to_csv(out / "local_ssc_scan.csv", index=False)
    eligible = scan[(scan.clusters >= 70) & (scan.clusters <= 500) & (scan.max_cluster_fraction < .08)]
    best = eligible.sort_values(["reference_hole_ami", "reference_hole_ari"], ascending=False).head(1)
    report = {
        "n_events": int(len(x)), "features": FEATURES,
        "method": "50-NN candidate graph in raw robust-scaled 5D; L1 affine sparse self-expression; weighted Louvain partition",
        "not_used_in_construction": ["foil_position", "cluster", "sieve_x", "sieve_y", "P_gtr_y"],
        "evaluation_only": "existing sieve-plane cluster and foil labels",
        "best_candidate": best.to_dict(orient="records")[0] if not best.empty else None,
        "verdict": "same-run exploratory diagnostic; use only a material improvement over local graph baselines as evidence for local SSC."
    }
    (out / "local_ssc_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
