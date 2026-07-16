"""Make annotated sieve and ytar figures matching the prior full-sample style.

All 'foil-like' bands are inferred post hoc from ytar. Existing foil labels
are never loaded.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from sklearn.mixture import GaussianMixture

OUT = Path(__file__).parent / "results"
COLORS = ["#8EA9C9", "#EDBD7C", "#8DCCB7"]


def make_band_maps(df, centers):
    gmm = GaussianMixture(n_components=3, random_state=25521, n_init=20).fit(centers[["ytar"]])
    order = np.argsort(gmm.means_.ravel()); remap = {old: new for new, old in enumerate(order)}
    centers["cluster_band"] = [remap[v] for v in gmm.predict(centers[["ytar"]])]
    event_band = np.array([remap[v] for v in gmm.predict(df[["P.gtr.y"]].to_numpy())])
    label_band = centers.cluster_band.to_dict()
    df["event_band"] = event_band
    df["cluster_band"] = df.flow_hdbscan_cluster.map(label_band)
    return centers, df


def colors_for_panel(part, centers):
    labels = np.sort(part.flow_hdbscan_cluster.unique())
    ordered = centers.loc[labels].sort_values(["sieve_y", "sieve_x"]).index.to_list()
    return {lab: hsv_to_rgb((i / max(len(ordered), 1), .74, .88)) for i, lab in enumerate(ordered)}


def draw_annotated_sieve(df, centers):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.2), constrained_layout=True, sharex=True, sharey=True)
    for band, ax in enumerate(axes):
        signal = df[(df.flow_hdbscan_cluster >= 0) & (df.cluster_band == band)]
        noise = df[(df.flow_hdbscan_cluster < 0) & (df.event_band == band)]
        colors = colors_for_panel(signal, centers)
        ax.scatter(noise.sieve_x, noise.sieve_y, c="0.80", s=.40, alpha=.20, linewidths=0, rasterized=True)
        ax.scatter(signal.sieve_x, signal.sieve_y, c=[colors[v] for v in signal.flow_hdbscan_cluster], s=.72, alpha=.76, linewidths=0, rasterized=True)
        local = centers[centers.cluster_band == band]
        ax.scatter(local.sieve_x, local.sieve_y, c="red", s=11, edgecolors="white", linewidths=.25, zorder=5)
        for cluster, row in local.iterrows():
            ax.annotate(str(cluster), (row.sieve_x, row.sieve_y), xytext=(2, 2), textcoords="offset points", color="red", fontsize=4.7, fontweight="bold", zorder=6)
        ytar = local.ytar.median()
        ax.set_title(f"Inferred foil-like band {band}  ($y_{{tar}}$={ytar:+.2f} cm)", fontsize=13)
        ax.set(xlim=(-16.5, 14.5), ylim=(-9, 9), xlabel=r"reconstructed $x_{sieve}$")
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"reconstructed $y_{sieve}$")
    fig.suptitle("Unskimmed ROOT: continuous-prior FP 5D flow + HDBSCAN projected to reconstructed sieve plane", fontsize=16.5, fontweight="bold")
    fig.savefig(OUT / "raw_fullroot_flow_hdbscan_annotated_three_band_sieve.png", dpi=280, bbox_inches="tight")


def draw_ytar_distribution(df, centers):
    bins = np.linspace(-3.1, 3.6, 55)
    fig, axes = plt.subplots(2, 1, figsize=(20, 11), constrained_layout=True, sharex=True)
    for band in range(3):
        p = df[df.event_band == band]
        axes[0].hist(p["P.gtr.y"], bins=bins, histtype="stepfilled", alpha=.60, color=COLORS[band],
                     label=f"inferred ytar band {band}: {len(p):,} events")
    axes[0].set(title="Reconstructed ytar distribution before using HDBSCAN labels", ylabel="events / bin")
    axes[0].legend(ncol=3, loc="upper center", frameon=False); axes[0].grid(alpha=.16)

    purity_rows = []
    for label, p in df[df.flow_hdbscan_cluster >= 0].groupby("flow_hdbscan_cluster"):
        assigned = int(centers.loc[label, "cluster_band"])
        purity_rows.append((label, assigned, float((p.event_band == assigned).mean()), len(p)))
    purity = pd.DataFrame(purity_rows, columns=["cluster", "band", "purity", "events"])
    for band in range(3):
        assigned = purity[purity.band == band].cluster
        p = df[df.flow_hdbscan_cluster.isin(assigned)]
        axes[1].hist(p["P.gtr.y"], bins=bins, histtype="stepfilled", alpha=.60, color=COLORS[band],
                     label=f"cluster-assigned band {band}: {len(assigned)} clusters, {len(p):,} events, mean purity {purity[purity.band == band].purity.mean():.3f}")
    noise = df[df.flow_hdbscan_cluster < 0]
    axes[1].hist(noise["P.gtr.y"], bins=bins, histtype="step", linewidth=1.4, color="0.35", label=f"HDBSCAN noise: {len(noise):,} events")
    axes[1].set(title="Post-clustering band assignment: each inferred cluster is assigned its median-ytar band", xlabel=r"reconstructed $y_{tar}$ ($P_{gtr,y}$)", ylabel="events / bin")
    axes[1].legend(ncol=2, loc="upper center", frameon=False); axes[1].grid(alpha=.16)
    fig.suptitle("Unskimmed ROOT FP 5D flow + HDBSCAN: foil-like structure along reconstructed ytar", fontsize=16.5, fontweight="bold")
    fig.savefig(OUT / "raw_fullroot_flow_hdbscan_ytar_band_distribution.png", dpi=260, bbox_inches="tight")
    return purity


if __name__ == "__main__":
    df = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_labels.csv")
    centers = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_centers.csv").set_index("flow_hdbscan_cluster")
    centers, df = make_band_maps(df, centers)
    draw_annotated_sieve(df, centers)
    purity = draw_ytar_distribution(df, centers)
    purity.to_csv(OUT / "raw_fullroot_flow_hdbscan_cluster_band_purity.csv", index=False)
