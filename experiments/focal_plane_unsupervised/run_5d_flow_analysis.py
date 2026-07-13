"""Diagnose foil/hole geometry in the *unreduced* five-dimensional focal-plane space.

The existing sieve-plane labels are read only after the 5D neighbourhoods have
been constructed.  They are a reference for interpreting the geometry, never
features of the 5D metric.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260713)
COLS = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
SHORT = ["xfp", "yfp", "xpfp", "ypfp", "fr_ybpm"]


def sample_per_group(frame, group, n):
    parts = []
    for _, g in frame.groupby(group, observed=True):
        parts.append(g.sample(min(n, len(g)), random_state=int(RNG.integers(2**31 - 1))))
    return pd.concat(parts, ignore_index=True)


def q(v):
    return float(np.round(v, 5))


def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=COLS + ["foil_position", "cluster", "sieve_x", "sieve_y"])
    # Equal foil sampling avoids letting the most populated foil set the metric.
    df = sample_per_group(df, "foil_position", 25000)
    scaler = RobustScaler().fit(df[COLS])
    x = scaler.transform(df[COLS])
    foil = df.foil_position.astype(int).to_numpy()
    hole = df.cluster.astype(int).to_numpy()
    n = len(df)

    nn = NearestNeighbors(n_neighbors=41, algorithm="auto").fit(x)
    dist, ind = nn.kneighbors(x)
    neigh_foil = foil[ind[:, 1:21]]
    neigh_hole = hole[ind[:, 1:21]]
    foil_purity = (neigh_foil == foil[:, None]).mean(axis=1)
    hole_purity = (neigh_hole == hole[:, None]).mean(axis=1)
    # Local PCA describes the tangent flow without changing the clustering space.
    local_dims, pc1, pc2, cond = [], [], [], []
    for ids in ind[:, :31]:
        eig = np.linalg.eigvalsh(np.cov(x[ids], rowvar=False))[::-1]
        frac = eig / max(eig.sum(), 1e-12)
        local_dims.append((eig.sum() ** 2) / max((eig**2).sum(), 1e-12))
        pc1.append(frac[0]); pc2.append(frac[1])
        cond.append(eig[0] / max(eig[2], 1e-12))
    local_dims, pc1, pc2, cond = map(np.asarray, (local_dims, pc1, pc2, cond))

    # Centroid distance relative to pooled radial size says whether foil sheets
    # are separate streams or overlapping folds in exactly the same 5D metric.
    foil_rows = []
    centers = {}
    radii = {}
    for f in sorted(np.unique(foil)):
        ids = np.flatnonzero(foil == f)
        centers[f] = x[ids].mean(axis=0)
        radii[f] = np.median(np.linalg.norm(x[ids] - centers[f], axis=1))
        foil_rows.append({"foil": int(f), "events": int(len(ids)),
                          "median_same_foil_k20": q(foil_purity[ids].mean()),
                          "median_same_hole_k20": q(hole_purity[ids].mean()),
                          "local_effective_dimension_median": q(np.median(local_dims[ids])),
                          "local_pc1_fraction_median": q(np.median(pc1[ids])),
                          "local_pc1_pc2_fraction_median": q(np.median(pc1[ids] + pc2[ids])),
                          "local_flow_condition_median": q(np.median(cond[ids]))})
    pairs = []
    fs = sorted(centers)
    for i, a in enumerate(fs):
        for b in fs[i+1:]:
            cd = np.linalg.norm(centers[a] - centers[b])
            pairs.append({"foils": f"{a}-{b}", "centroid_distance": q(cd),
                          "distance_over_mean_radius": q(cd / ((radii[a]+radii[b])/2))})

    # Same-foil cluster/hole flow: how much a labelled hole stays a coherent 5D tube.
    holes = []
    for (f, h), g in df.groupby(["foil_position", "cluster"], observed=True):
        ids = g.index.to_numpy()
        # indices in sampled dataframe are reset, but group index remains contiguous.
        if len(ids) < 40:
            continue
        holes.append({"foil": int(f), "cluster": int(h), "events": int(len(ids)),
                      "same_hole_k20": q(hole_purity[ids].mean()),
                      "same_foil_k20": q(foil_purity[ids].mean()),
                      "local_dim": q(np.median(local_dims[ids])),
                      "median_nn5_distance": q(np.median(dist[ids, 5]))})
    holes_df = pd.DataFrame(holes)
    holes_df.to_csv(OUT / "5d_hole_flow_by_reference_cluster.csv", index=False)

    # Silhouette is only sampled for computational cost, using labels purely as post-hoc reference.
    take = RNG.choice(n, min(12000, n), replace=False)
    sil_foil = silhouette_score(x[take], foil[take])
    # Compare only holes inside each foil; global labels artificially conflate foils.
    sil_holes = {}
    for f in fs:
        ids = np.flatnonzero(foil == f)
        choose = RNG.choice(ids, min(7000, len(ids)), replace=False)
        sil_holes[str(f)] = q(silhouette_score(x[choose], hole[choose]))

    summary = {
        "space": SHORT,
        "n_events": int(n),
        "metric": "global robust-scaled Euclidean, no embedding or target coordinates",
        "foil_silhouette": q(sil_foil),
        "foil_pairs": pairs,
        "per_foil": foil_rows,
        "hole_silhouette_within_foil": sil_holes,
        "all_events_k20_same_foil": q(foil_purity.mean()),
        "all_events_k20_same_reference_hole": q(hole_purity.mean()),
        "local_effective_dimension_median": q(np.median(local_dims)),
        "local_effective_dimension_iqr": [q(np.quantile(local_dims,.25)), q(np.quantile(local_dims,.75))],
        "local_pc1_plus_pc2_median": q(np.median(pc1+pc2)),
        "note": "foil_position and cluster were not used to build neighbours; they only interpret the resulting 5D geometry."
    }
    (OUT / "5d_flow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Coordinate views: no projection of the five-dimensional metric, only direct coordinate slices.
    plot = sample_per_group(df, "foil_position", 3500)
    px = scaler.transform(plot[COLS])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    pairs_plot = [(0,2), (1,3), (0,1), (2,3), (0,4), (1,4)]
    colors = {0:"#3366cc", 1:"#dc3912", 2:"#109618"}
    for ax, (a,b) in zip(axes.flat, pairs_plot):
        for f in fs:
            m = plot.foil_position.to_numpy().astype(int) == f
            ax.scatter(px[m,a], px[m,b], s=1.0, alpha=.28, color=colors[f], label=f"foil {f}")
        ax.set_xlabel(SHORT[a] + " (robust scaled)")
        ax.set_ylabel(SHORT[b] + " (robust scaled)")
    axes[0,0].legend(markerscale=5, frameon=False)
    fig.savefig(OUT / "09_5d_coordinate_slices_by_foil.png", dpi=180)
    plt.close(fig)

    # Distribution of post-hoc local purity exposes the continuous / folded regions.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for f in fs:
        ids = foil == f
        axes[0].hist(foil_purity[ids], bins=30, density=True, histtype="step", lw=2, color=colors[f], label=f"foil {f}")
        axes[1].hist(hole_purity[ids], bins=30, density=True, histtype="step", lw=2, color=colors[f])
        axes[2].hist(local_dims[ids], bins=30, density=True, histtype="step", lw=2, color=colors[f])
    axes[0].set(title="20-NN foil purity", xlabel="fraction with same foil", ylabel="density")
    axes[1].set(title="20-NN hole purity", xlabel="fraction with same reference hole")
    axes[2].set(title="local effective dimension (30-NN PCA)", xlabel="participation ratio")
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "10_5d_neighborhood_flow_diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
