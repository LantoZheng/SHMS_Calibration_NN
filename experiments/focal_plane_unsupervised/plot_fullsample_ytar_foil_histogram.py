"""Show post-clustering foil assignment along reconstructed ytar for the full-sample display."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
DATA = OUT / "continuous_prior_flow_fullsample_labels.csv"


def main():
    df = pd.read_csv(DATA)
    active = df.flow_hdbscan_cluster >= 0
    signal = df[active].copy()
    # Foil is never used by the flow or HDBSCAN.  This is solely a posterior
    # diagnostic: assign each discovered cluster the foil containing most of it.
    cluster_foil = signal.groupby("flow_hdbscan_cluster").foil_position.agg(lambda x: x.mode().iat[0]).astype(int)
    cluster_purity = signal.groupby("flow_hdbscan_cluster").foil_position.apply(lambda x: x.value_counts(normalize=True).iat[0])
    signal["inferred_foil"] = signal.flow_hdbscan_cluster.map(cluster_foil)
    signal["cluster_foil_purity"] = signal.flow_hdbscan_cluster.map(cluster_purity)
    df["inferred_foil"] = df.flow_hdbscan_cluster.map(cluster_foil)
    df.to_csv(OUT / "continuous_prior_flow_fullsample_foil_assignment.csv", index=False)

    foils = sorted(signal.foil_position.unique())
    colors = {foil: color for foil, color in zip(foils, ["#386cb0", "#e08214", "#1b9e77"])}
    bins = np.linspace(df.P_gtr_y.quantile(.002), df.P_gtr_y.quantile(.998), 76)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.3), sharex=True, constrained_layout=True)
    for foil in foils:
        part = df[df.foil_position == foil]
        axes[0].hist(part.P_gtr_y, bins=bins, histtype="stepfilled", alpha=.30, color=colors[foil], label=f"reference foil {foil}: {len(part):,} events")
    axes[0].set_ylabel("events / bin")
    axes[0].set_title(r"Reconstructed $y_{tar}$ distribution before using cluster labels", fontweight="bold")
    axes[0].legend(ncol=3, fontsize=9, frameon=False)
    axes[0].grid(alpha=.18)

    for foil in foils:
        part = signal[signal.inferred_foil == foil]
        n_clusters = part.flow_hdbscan_cluster.nunique()
        purity = part.cluster_foil_purity.mean()
        axes[1].hist(part.P_gtr_y, bins=bins, histtype="stepfilled", alpha=.35, color=colors[foil], label=f"cluster-assigned foil {foil}: {n_clusters} clusters, {len(part):,} events, mean purity {purity:.3f}")
    noise = df[~active]
    axes[1].hist(noise.P_gtr_y, bins=bins, histtype="step", linewidth=1.15, color="0.38", label=f"HDBSCAN noise: {len(noise):,} events")
    axes[1].set_xlabel(r"reconstructed $y_{tar}$  ($P_{gtr,y}$)")
    axes[1].set_ylabel("events / bin")
    axes[1].set_title(r"Post-clustering foil assignment: each inferred cluster is assigned its majority foil", fontweight="bold")
    axes[1].legend(ncol=2, fontsize=8.6, frameon=False)
    axes[1].grid(alpha=.18)
    fig.suptitle(r"Full-sample FP 5D flow + HDBSCAN: foil structure along $y_{tar}$", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "18_continuous_prior_flow_ytar_foil_histogram.png", dpi=240, bbox_inches="tight")
    print({"clusters": int(len(cluster_foil)), "noise_fraction": round(float((~active).mean()), 4), "cluster_mean_majority_foil_purity": round(float(cluster_purity.mean()), 4)})


if __name__ == "__main__":
    main()
