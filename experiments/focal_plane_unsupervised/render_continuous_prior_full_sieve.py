"""Full-sample qualitative view of all sieve holes after continuous-prior FP5D flow clustering.

This is a geometry/coverage visualization only: it fits on the complete sampled
run and must not be read as an independent held-out performance measurement.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors as mcolors
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from run_continuous_prior_flow_metric import FEATURES, OPTICAL, RASTER, SEED, TARGET, Flow

OUT = Path(__file__).parent / "results"
DATA = Path(__file__).resolve().parents[2] / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"


def spatially_contrasting_colors(frame):
    """Greedy colour assignment: closest inferred clusters never share a hue."""
    centers = frame.groupby("flow_hdbscan_cluster")[["sieve_x", "sieve_y"]].mean()
    labels = centers.index.to_numpy()
    xy = centers.to_numpy()
    distance = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(axis=2))
    # 36 deliberately separated hues; red is reserved for centroid markers.
    palette = [mcolors.hsv_to_rgb(((h % 18) / 18, .78 if h < 18 else .56, .92 if h < 18 else .72)) for h in range(36)]
    order = np.argsort(np.min(np.where(distance > 0, distance, np.inf), axis=1))
    assigned = {}
    for idx in order:
        near = np.argsort(distance[idx])[1:13]
        used = {assigned[labels[j]] for j in near if labels[j] in assigned}
        assigned[labels[idx]] = next(c for c in range(len(palette)) if c not in used)
    return {lab: palette[color] for lab, color in assigned.items()}, centers


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = pd.read_csv(DATA, usecols=FEATURES + TARGET + ["foil_position"]).dropna()
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.choice(len(df), 60000, replace=False)].reset_index(drop=True)
    condition = make_pipeline(SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), Ridge(alpha=2.0))
    condition.fit(df[[RASTER]], df[OPTICAL])
    corrected = np.column_stack([df[OPTICAL].to_numpy() - condition.predict(df[[RASTER]]), df[RASTER].to_numpy()])
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(corrected).astype("float32")
    y = StandardScaler().fit_transform(df[TARGET]).astype("float32")
    model = Flow(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=512, shuffle=True)
    for _ in range(100):
        for xb, yb in loader:
            z = model(xb)
            loss = ((z[:, :3] - yb) ** 2).mean() + .02 * ((z[:, 3:] - xb[:, 3:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
    with torch.no_grad(): z = model(torch.from_numpy(x)).numpy()
    z[:, 3:] *= np.sqrt(.10)
    pred = HDBSCAN(min_cluster_size=15, min_samples=5, cluster_selection_method="eom").fit_predict(z)
    df["flow_hdbscan_cluster"] = pred
    df.to_csv(OUT / "continuous_prior_flow_fullsample_labels.csv", index=False)

    active = pred >= 0
    colors, centers = spatially_contrasting_colors(df[df.flow_hdbscan_cluster >= 0])
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4), constrained_layout=True, sharex=True, sharey=True)
    for ax, foil in zip(axes, sorted(df.foil_position.unique())):
        part = df[df.foil_position == foil]
        signal = part[part.flow_hdbscan_cluster >= 0]; noise = part[part.flow_hdbscan_cluster < 0]
        ax.scatter(signal.sieve_x, signal.sieve_y, c=[colors[c] for c in signal.flow_hdbscan_cluster], s=1.10, alpha=.78, linewidths=0, rasterized=True)
        ax.scatter(noise.sieve_x, noise.sieve_y, c="0.68", s=.60, alpha=.22, linewidths=0, rasterized=True)
        local_centers = centers.loc[centers.index.isin(signal.flow_hdbscan_cluster.unique())]
        ax.scatter(local_centers.sieve_x, local_centers.sieve_y, c="red", s=13, marker="o", edgecolors="white", linewidths=.35, zorder=5)
        for label, row in local_centers.iterrows():
            ax.annotate(str(label), (row.sieve_x, row.sieve_y), xytext=(2.5, 2.5), textcoords="offset points",
                        color="red", fontsize=4.8, fontweight="bold", zorder=6,
                        bbox={"boxstyle": "round,pad=.08", "fc": "white", "ec": "none", "alpha": .62})
        ax.set_title(f"Foil {foil}")
        ax.set_xlabel(r"reconstructed $x_{sieve}$")
        ax.grid(alpha=.16)
    axes[0].set_ylabel(r"reconstructed $y_{sieve}$")
    fig.suptitle("All sampled events: continuous-prior FP 5D flow + HDBSCAN projected to reconstructed sieve plane", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "17_continuous_prior_flow_full_sieve_all_holes.png", dpi=260, bbox_inches="tight")
    print({"events": len(df), "clusters": int(len(np.unique(pred[active]))), "noise_fraction": float((~active).mean())})


if __name__ == "__main__":
    main()
