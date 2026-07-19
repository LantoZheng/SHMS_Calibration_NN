"""Static figures used by the PI progress report."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"


def comparison():
    names = ["Raw 5D\nHDBSCAN", "Conditioned\nraw 5D", "Continuous flow\n5D equal", "Continuous flow\n5D residual 0.10", "Discrete-center\nflow reference"]
    ami = [0.230, 0.230, 0.947, 0.952, 0.958]
    ari = [-0.002, -0.002, 0.904, 0.916, 0.928]
    noise = [0.710, 0.710, 0.051, 0.046, 0.033]
    clusters = [128, 128, 46, 47, 46]
    colors = ["#8c8c8c", "#6c8ebf", "#3a7d44", "#1f5f8b", "#b07d2b"]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)
    axes[0].bar(x-.18, ami, .36, label="AMI", color=colors, alpha=.95)
    axes[0].bar(x+.18, ari, .36, label="ARI", color=colors, alpha=.48, hatch="//")
    axes[0].set(ylim=(-.08, 1.05), ylabel="Agreement with sieve-plane reference", xticks=x, xticklabels=names)
    axes[0].axhline(0, color="0.45", lw=.8); axes[0].legend(frameon=False, loc="upper left")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar(x, noise, color=colors)
    for xi, n, v in zip(x, clusters, noise): axes[1].text(xi, min(.78, v+.025), f"{n} clusters", ha="center", va="bottom", fontsize=8)
    axes[1].set(ylim=(0,.82), ylabel="HDBSCAN noise fraction", xticks=x, xticklabels=names)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].set_title("Same 44-hole held-out test split; gold is stronger-supervision reference", fontsize=9)
    fig.savefig(OUT / "14_pi_performance_comparison.png", dpi=220)
    plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(12.4, 3.8))
    ax.set_axis_off(); ax.set(xlim=(0, 1), ylim=(0, 1))
    steps = [
        (0.03, "Raw focal-plane event\nxfp, yfp, xpfp, ypfp, fr_ybpm", "#E8EEF5"),
        (0.27, "Train-only raster conditioning\noptical residuals + fr_ybpm", "#EAF3EA"),
        (0.51, "Invertible RealNVP\n5D -> 5D transport", "#EAF3EA"),
        (0.75, "Weighted full-5D distance\nthen HDBSCAN", "#E8EEF5"),
    ]
    for x, text, fill in steps:
        box = plt.Rectangle((x,.32), .19,.36, facecolor=fill, edgecolor="#31506D", lw=1.5)
        ax.add_patch(box); ax.text(x+.095,.50,text,ha="center",va="center",fontsize=10)
    for x in (.22,.46,.70): ax.annotate("", xy=(x+.045,.50), xytext=(x,.50), arrowprops=dict(arrowstyle="->",lw=1.5,color="#31506D"))
    ax.text(.605,.12,"Weak continuous targets during training only: sieve_x, sieve_y, P_gtr_y\nNot used: cluster, hole ID, cluster centers, foil labels", ha="center",va="center",fontsize=10,color="#29475F")
    ax.text(.035,.82,"Hold out complete reference holes before any fitting; use them only for final evaluation.", fontsize=10, color="#29475F")
    fig.savefig(OUT / "15_continuous_prior_flow_schematic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    comparison(); schematic()
