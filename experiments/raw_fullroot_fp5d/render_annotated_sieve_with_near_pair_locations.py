"""Overlay the close-sieve cluster pairs on the annotated three-band figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).parent / "results"
PAIR_SPEC = [("P1", 219, 220, "#222222"), ("P2", 97, 110, "#D55E00"), ("P3", 128, 150, "#7B3294"), ("P4", 100, 107, "#009E73")]


def panel_colors(labels, centers):
    ordered = centers.loc[labels].sort_values(["sieve_y", "sieve_x"]).index.to_list()
    return {lab: hsv_to_rgb((i / max(len(ordered), 1), .74, .88)) for i, lab in enumerate(ordered)}


def main():
    df = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_labels.csv")
    centers = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_centers.csv").set_index("flow_hdbscan_cluster")
    gmm = GaussianMixture(n_components=3, random_state=25521, n_init=20).fit(centers[["ytar"]])
    order = np.argsort(gmm.means_.ravel()); remap = {old: new for new, old in enumerate(order)}
    centers["band"] = [remap[v] for v in gmm.predict(centers[["ytar"]])]
    df["event_band"] = [remap[v] for v in gmm.predict(df[["P.gtr.y"]].to_numpy())]
    df["cluster_band"] = df.flow_hdbscan_cluster.map(centers.band.to_dict())

    fig, axes = plt.subplots(1, 3, figsize=(20, 7.6), constrained_layout=True, sharex=True, sharey=True)
    for band, ax in enumerate(axes):
        signal = df[(df.flow_hdbscan_cluster >= 0) & (df.cluster_band == band)]
        noise = df[(df.flow_hdbscan_cluster < 0) & (df.event_band == band)]
        colors = panel_colors(signal.flow_hdbscan_cluster.unique(), centers)
        ax.scatter(noise.sieve_x, noise.sieve_y, c="0.80", s=.40, alpha=.18, linewidths=0, rasterized=True)
        ax.scatter(signal.sieve_x, signal.sieve_y, c=[colors[v] for v in signal.flow_hdbscan_cluster], s=.72, alpha=.73, linewidths=0, rasterized=True)
        local = centers[centers.band == band]
        ax.scatter(local.sieve_x, local.sieve_y, c="red", s=9, edgecolors="white", linewidths=.2, zorder=4)
        for cluster, row in local.iterrows():
            ax.annotate(str(cluster), (row.sieve_x, row.sieve_y), xytext=(2, 2), textcoords="offset points", color="red", fontsize=4.4, fontweight="bold", zorder=5)
        ytar = local.ytar.median()
        ax.set_title(f"Inferred foil-like band {band}  ($y_{{tar}}$={ytar:+.2f} cm)", fontsize=13)
        ax.set(xlim=(-16.5, 14.5), ylim=(-9, 9), xlabel=r"reconstructed $x_{sieve}$")
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"reconstructed $y_{sieve}$")

    # Highlight pair members in the panel(s) where each cluster resides.
    for code, a, b, color in PAIR_SPEC:
        ra, rb = centers.loc[a], centers.loc[b]
        for cluster, row in ((a, ra), (b, rb)):
            ax = axes[int(row.band)]
            ax.add_patch(Circle((row.sieve_x, row.sieve_y), radius=.46, fill=False, edgecolor=color, linewidth=2.25, zorder=7))
            ax.annotate(f"{code}:{cluster}", (row.sieve_x, row.sieve_y), xytext=(7, -11), textcoords="offset points", color=color, fontsize=7.6, fontweight="bold", zorder=8,
                        bbox={"boxstyle": "round,pad=.12", "fc": "white", "ec": color, "alpha": .88, "lw": .6})
        if int(ra.band) == int(rb.band):
            axes[int(ra.band)].plot([ra.sieve_x, rb.sieve_x], [ra.sieve_y, rb.sieve_y], color=color, linewidth=1.6, zorder=6)
    legend = [Line2D([0], [0], color=c, marker="o", markerfacecolor="none", markeredgewidth=1.8, label=f"{code}: clusters {a} ↔ {b}") for code, a, b, c in PAIR_SPEC]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, -.015))
    fig.suptitle("Near-sieve cluster pairs highlighted: members may lie in different inferred foil-like ytar bands", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "near_sieve_cluster_pairs_on_annotated_sieve.png", dpi=280, bbox_inches="tight")


if __name__ == "__main__": main()
