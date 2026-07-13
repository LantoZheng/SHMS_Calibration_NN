"""Project the selected tangent-aware graph partition back onto measured FP and sieve coordinates."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_tangent_multimanifold_graph as model


def main():
    out = HERE / "results"
    df = model.load()
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[model.FEATURES])
    rows, cols, mahal, tang, sigma = model.build_geometry(x)
    labels, _ = model.graph_labels(len(x), rows, cols, mahal, tang, sigma, tangent_weight=6.0, resolution=8.0)
    df = df.copy()
    df["tangent_graph_cluster"] = labels
    ref = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    # Per-community reference purity is only a diagnostic; it was not used in clustering.
    purity = np.empty(len(df))
    for c in np.unique(labels):
        mask = labels == c
        _, counts = np.unique(ref[mask], return_counts=True)
        purity[mask] = counts.max() / mask.sum()
    df["reference_cluster_purity"] = purity
    df.to_csv(out / "tangent_multimanifold_graph_labels.csv", index=False)

    # A stable discrete hue map makes the same graph community traceable across all panels.
    cmap = plt.colormaps["hsv"]
    colors = cmap((labels % 81) / 81.0)
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    panels = [
        ("P_dc_x_fp", "P_dc_xp_fp", "FP: xfp vs xpfp"),
        ("P_dc_y_fp", "P_dc_yp_fp", "FP: yfp vs ypfp"),
        ("sieve_x", "sieve_y", "sieve plane: tangent-graph community"),
    ]
    for a, (u, v, title) in zip(ax.flat[:3], panels):
        a.scatter(df[u], df[v], s=2.2, c=colors, alpha=.62, linewidths=0, rasterized=True)
        a.set(xlabel=u, ylabel=v, title=title)
    s = ax[1, 1].scatter(df.sieve_x, df.sieve_y, s=2.2, c=purity, cmap="viridis", vmin=0, vmax=1, alpha=.7, linewidths=0, rasterized=True)
    ax[1, 1].set(xlabel="sieve_x", ylabel="sieve_y", title="sieve plane: graph-community reference purity")
    fig.colorbar(s, ax=ax[1, 1], label="dominant sieve-reference fraction in graph community")
    fig.savefig(out / "11_tangent_graph_fp_sieve_projection.png", dpi=200)
    plt.close(fig)
    print({"events": len(df), "communities": int(labels.max()+1), "median_purity": float(np.median(purity))})


if __name__ == "__main__":
    main()
