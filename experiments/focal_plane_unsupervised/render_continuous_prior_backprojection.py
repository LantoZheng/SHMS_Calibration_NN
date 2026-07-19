"""Reproduce the strict hole-holdout flow result and plot cluster labels back in reconstructed coordinates."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from run_continuous_prior_flow_metric import FEATURES, OPTICAL, RASTER, SEED, TARGET, Flow


OUT = Path(__file__).parent / "results"
DATA = Path(__file__).resolve().parents[2] / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    use = FEATURES + TARGET + ["foil_position", "cluster"]
    df = pd.read_csv(DATA, usecols=use).dropna()
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.choice(len(df), 60000, replace=False)].reset_index(drop=True)
    ref = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    held = rng.choice(np.unique(ref), size=int(np.ceil(.2 * len(np.unique(ref)))), replace=False)
    test = np.isin(ref, held); train = ~test

    condition = make_pipeline(SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), Ridge(alpha=2.0))
    condition.fit(df.loc[train, [RASTER]], df.loc[train, OPTICAL])
    corrected = np.column_stack([df[OPTICAL].to_numpy() - condition.predict(df[[RASTER]]), df[RASTER].to_numpy()])
    xs = RobustScaler(quantile_range=(5, 95)).fit(corrected[train])
    ys = StandardScaler().fit(df.loc[train, TARGET])
    x = xs.transform(corrected).astype("float32")
    y = ys.transform(df[TARGET]).astype("float32")

    model = Flow(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])), batch_size=512, shuffle=True)
    for _ in range(100):
        for xb, yb in loader:
            z = model(xb)
            loss = ((z[:, :3] - yb) ** 2).mean() + .02 * ((z[:, 3:] - xb[:, 3:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
    with torch.no_grad(): z = model(torch.from_numpy(x)).numpy()
    weighted = z.copy(); weighted[:, 3:] *= np.sqrt(.10)
    pred = HDBSCAN(min_cluster_size=15, min_samples=5, cluster_selection_method="eom").fit_predict(weighted[test])

    plot = df.loc[test, ["sieve_x", "sieve_y", "P_gtr_y", "foil_position", "cluster"]].copy().reset_index(drop=True)
    plot["flow_hdbscan_cluster"] = pred
    plot["is_noise"] = pred < 0
    plot.to_csv(OUT / "continuous_prior_flow_holeholdout_backprojection_labels.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4), constrained_layout=True, sharex="row")
    cmap = plt.get_cmap("turbo")
    for col, foil in enumerate(sorted(plot.foil_position.unique())):
        part = plot[plot.foil_position == foil]
        signal = part[~part.is_noise]
        noise = part[part.is_noise]
        # Cluster index is only a categorical display color; no reference label controls the color.
        axes[0, col].scatter(signal.sieve_x, signal.sieve_y, c=signal.flow_hdbscan_cluster, cmap=cmap, s=2.2, alpha=.72, linewidths=0)
        axes[0, col].scatter(noise.sieve_x, noise.sieve_y, c="0.70", s=1.7, alpha=.35, linewidths=0, label="HDBSCAN noise")
        axes[0, col].set_title(f"Foil {foil}: reconstructed sieve plane")
        axes[0, col].set_xlabel(r"reconstructed $x_{sieve}$")
        axes[0, col].set_ylabel(r"reconstructed $y_{sieve}$")
        axes[0, col].grid(alpha=.18)
        axes[1, col].scatter(signal.sieve_y, signal.P_gtr_y, c=signal.flow_hdbscan_cluster, cmap=cmap, s=2.2, alpha=.72, linewidths=0)
        axes[1, col].scatter(noise.sieve_y, noise.P_gtr_y, c="0.70", s=1.7, alpha=.35, linewidths=0)
        axes[1, col].set_title(f"Foil {foil}: target-y consistency")
        axes[1, col].set_xlabel(r"reconstructed $y_{sieve}$")
        axes[1, col].set_ylabel(r"reconstructed $y_{tar}$ ($P_{gtr,y}$)")
        axes[1, col].grid(alpha=.18)
    fig.suptitle("Continuous-prior FP 5D flow + HDBSCAN: held-out-hole labels projected back to reconstruction", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "16_continuous_prior_flow_backprojection_sieve_ytar.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
