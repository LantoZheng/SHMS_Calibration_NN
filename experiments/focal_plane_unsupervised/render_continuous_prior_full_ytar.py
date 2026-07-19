"""Foil-resolved ytar-direction projection of the saved full-sample clustering."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from render_continuous_prior_full_sieve import spatially_contrasting_colors

OUT = Path(__file__).parent / "results"


def main():
    df = pd.read_csv(OUT / "continuous_prior_flow_fullsample_labels.csv")
    active = df.flow_hdbscan_cluster >= 0
    # Reuse the spatially contrastive cluster colours from the sieve-plane view.
    colors, _ = spatially_contrasting_colors(df.loc[active])
    centers = df.loc[active].groupby(["foil_position", "flow_hdbscan_cluster"])[["sieve_x", "P_gtr_y"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4), constrained_layout=True, sharex=True)
    for ax, foil in zip(axes, sorted(df.foil_position.unique())):
        part = df[df.foil_position == foil]
        signal = part[part.flow_hdbscan_cluster >= 0]; noise = part[part.flow_hdbscan_cluster < 0]
        ax.scatter(signal.sieve_x, signal.P_gtr_y, c=[colors[c] for c in signal.flow_hdbscan_cluster], s=1.10, alpha=.78, linewidths=0, rasterized=True)
        ax.scatter(noise.sieve_x, noise.P_gtr_y, c="0.68", s=.60, alpha=.22, linewidths=0, rasterized=True)
        local = centers.loc[foil]
        ax.scatter(local.sieve_x, local.P_gtr_y, c="red", s=13, marker="o", edgecolors="white", linewidths=.35, zorder=5)
        for label, row in local.iterrows():
            ax.annotate(str(label), (row.sieve_x, row.P_gtr_y), xytext=(2.5, 2.5), textcoords="offset points",
                        color="red", fontsize=4.8, fontweight="bold", zorder=6,
                        bbox={"boxstyle": "round,pad=.08", "fc": "white", "ec": "none", "alpha": .62})
        ax.set_title(f"Foil {foil}")
        ax.set_xlabel(r"reconstructed $x_{sieve}$")
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"reconstructed $y_{tar}$ ($P_{gtr,y}$)")
    fig.suptitle("All sampled events: foil-resolved FP-flow clusters along reconstructed ytar", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "18_continuous_prior_flow_full_foil_ytar_projection.png", dpi=260, bbox_inches="tight")


if __name__ == "__main__":
    main()
