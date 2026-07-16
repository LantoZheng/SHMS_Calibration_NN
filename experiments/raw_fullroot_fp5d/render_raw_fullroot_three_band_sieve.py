"""Three-panel sieve-plane projection in the style of the previous foil figure.

Panels are *post-hoc* ytar bands inferred from the new clusters themselves,
not historical foil labels.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).parent / "results"


def panel_colors(labels, centers):
    order = centers.loc[labels].sort_values(["sieve_y", "sieve_x"]).index.to_list()
    hues = np.linspace(0, 1, max(24, len(order)), endpoint=False)
    return {lab: hsv_to_rgb((hues[i], .74, .88)) for i, lab in enumerate(order)}


def main():
    df = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_labels.csv")
    centers = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_centers.csv").set_index("flow_hdbscan_cluster")
    gmm = GaussianMixture(n_components=3, random_state=25521, n_init=20).fit(centers[["ytar"]])
    order = np.argsort(gmm.means_.ravel()); remap = {old: new for new, old in enumerate(order)}
    centers["band"] = [remap[x] for x in gmm.predict(centers[["ytar"]])]
    cluster_band = centers.band.to_dict()
    df["ytar_band"] = df.flow_hdbscan_cluster.map(cluster_band)
    # Noise receives a post-hoc continuous ytar-band membership only for display.
    noise = df.flow_hdbscan_cluster < 0
    df.loc[noise, "ytar_band"] = [remap[x] for x in gmm.predict(df.loc[noise, ["P.gtr.y"]].to_numpy())]
    df.ytar_band = df.ytar_band.astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), constrained_layout=True, sharex=True, sharey=True)
    for band, ax in enumerate(axes):
        part = df[df.ytar_band == band]
        signal = part[part.flow_hdbscan_cluster >= 0]
        bg = part[part.flow_hdbscan_cluster < 0]
        label_set = signal.flow_hdbscan_cluster.unique()
        colors = panel_colors(label_set, centers)
        ax.scatter(bg.sieve_x, bg.sieve_y, c="0.78", s=.42, alpha=.22, linewidths=0, rasterized=True)
        ax.scatter(signal.sieve_x, signal.sieve_y, c=[colors[c] for c in signal.flow_hdbscan_cluster], s=.75, alpha=.78, linewidths=0, rasterized=True)
        center = centers.loc[centers.band == band, "ytar"].median()
        ax.set_title(f"Inferred $y_{{tar}}$ band {band}  ({center:+.2f} cm)", fontsize=14)
        ax.set_xlabel(r"reconstructed $x_{sieve}$", fontsize=13)
        ax.set_xlim(-16.5, 14.5); ax.set_ylim(-9.0, 9.0)
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"reconstructed $y_{sieve}$", fontsize=13)
    fig.suptitle("Unskimmed ROOT: continuous-prior FP 5D flow + HDBSCAN projected to reconstructed sieve plane", fontsize=17, fontweight="bold")
    fig.savefig(OUT / "raw_fullroot_flow_hdbscan_three_ytar_band_sieve.png", dpi=260, bbox_inches="tight")


if __name__ == "__main__": main()
