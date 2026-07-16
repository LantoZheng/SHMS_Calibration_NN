"""Post-hoc structure audit for the label-free full-ROOT clustering result."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


OUT = Path(__file__).parent / "results"


def main():
    centers = pd.read_csv(OUT / "raw_fullroot_flow_hdbscan_centers.csv")
    # This is a display/audit-only fit on inferred cluster centres.  It is not
    # used by the FP flow or HDBSCAN and it does not read an existing foil label.
    gmm = GaussianMixture(n_components=3, random_state=25521, n_init=20).fit(centers[["ytar"]])
    bands = gmm.predict(centers[["ytar"]])
    order = np.argsort(gmm.means_.ravel())
    remap = {old: new for new, old in enumerate(order)}
    centers["ytar_band"] = [remap[x] for x in bands]
    centers["band_mean_ytar"] = centers.ytar_band.map({remap[i]: float(gmm.means_[i,0]) for i in range(3)})
    centers.to_csv(OUT / "raw_fullroot_flow_hdbscan_centers_with_ytar_bands.csv", index=False)

    summary = {"posthoc_only": "three-component GMM of inferred-cluster median ytar; no foil label used in flow or HDBSCAN",
               "bands": []}
    for band, piece in centers.groupby("ytar_band", sort=True):
        summary["bands"].append({"band": int(band), "cluster_count": int(len(piece)), "median_ytar": float(piece.ytar.median()),
                                 "ytar_q16_q84": [float(x) for x in piece.ytar.quantile([.16,.84])],
                                 "median_cluster_events": float(piece.events.median()),
                                 "sieve_x_range": [float(piece.sieve_x.min()), float(piece.sieve_x.max())],
                                 "sieve_y_range": [float(piece.sieve_y.min()), float(piece.sieve_y.max())]})
    (OUT / "raw_fullroot_flow_hdbscan_band_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = ["#376FA6", "#E4873C", "#4F9862"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True, sharex=True, sharey=True)
    for band, ax in enumerate(axes):
        part = centers[centers.ytar_band == band]
        size = np.clip(part.events / 7, 14, 210)
        ax.scatter(part.sieve_x, part.sieve_y, s=size, c=colors[band], alpha=.80, edgecolors="black", linewidths=.30)
        ax.set(title=f"post-hoc ytar band {band}: {part.ytar.median():+.2f} cm\n{len(part)} inferred clusters", xlabel="median reconstructed sieve_x")
        ax.grid(alpha=.16)
    axes[0].set_ylabel("median reconstructed sieve_y")
    fig.suptitle("All-foil structure discovered from raw FP5D clustering: three ytar bands each retain a sieve-hole lattice", fontsize=13.5, fontweight="bold")
    fig.savefig(OUT / "raw_fullroot_flow_hdbscan_ytar_band_lattices.png", dpi=230, bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
