"""Highlight closest sieve-plane pairs constrained to a shared inferred ytar band."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).parent / "results"
# Closest pair in band 0, closest in band 1, then two closest independent pairs in band 2.
PAIR_SPEC = [("S1", 219, 220, "#222222"), ("S2", 15, 68, "#D55E00"), ("S3", 2, 3, "#7B3294"), ("S4", 45, 56, "#009E73")]


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

    for code, a, b, color in PAIR_SPEC:
        ra, rb = centers.loc[a], centers.loc[b]
        assert int(ra.band) == int(rb.band), f"{a}, {b} are not in one ytar band"
        ax = axes[int(ra.band)]
        for cluster, row in ((a, ra), (b, rb)):
            ax.add_patch(Circle((row.sieve_x, row.sieve_y), radius=.46, fill=False, edgecolor=color, linewidth=2.25, zorder=7))
            ax.annotate(f"{code}:{cluster}", (row.sieve_x, row.sieve_y), xytext=(7, -11), textcoords="offset points", color=color, fontsize=7.6, fontweight="bold", zorder=8,
                        bbox={"boxstyle": "round,pad=.12", "fc": "white", "ec": color, "alpha": .88, "lw": .6})
        ax.plot([ra.sieve_x, rb.sieve_x], [ra.sieve_y, rb.sieve_y], color=color, linewidth=1.8, zorder=6)
    legend = [Line2D([0], [0], color=c, marker="o", markerfacecolor="none", markeredgewidth=1.8, label=f"{code}: {a} ↔ {b} (same ytar band)") for code, a, b, c in PAIR_SPEC]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(.5, -.015))
    fig.suptitle("Close or overlapping sieve-plane clusters constrained to the same inferred foil-like ytar band", fontsize=15.5, fontweight="bold")
    fig.savefig(OUT / "same_ytar_band_near_cluster_pairs_on_sieve.png", dpi=280, bbox_inches="tight")


if __name__ == "__main__": main()
