"""Cluster every usable event in the unskimmed run-25521 ROOT tree.

This experiment deliberately starts from the 521,282-entry replay ROOT file,
not from a stage-2 CSV or a skim.  It makes only finite-value, DC-sentinel,
kinematic and PID quality cuts; it never cuts on sieve coordinates or reads
foil/hole/cluster labels.  Continuous reconstructed sieve_x/sieve_y and ytar
are weak training targets, so this is a reconstruction-assisted (not purely
unsupervised) full-data clustering experiment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import uproot
from matplotlib.colors import hsv_to_rgb
from sklearn.cluster import HDBSCAN
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RAW = Path(os.environ.get(
    "SHMS_RUN25521_ROOT",
    Path(__file__).resolve().parents[3] / "RootData" / "shms_coin_replay_production_25521_-1.root",
)).expanduser().resolve()
OUT = Path(__file__).parent / "results"
SEED = 25521
FP = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp"]
RASTER = "P.rb.raster.frybRawAdc"
RECO = ["P.gtr.dp", "P.gtr.th", "P.gtr.ph", "P.gtr.x", "P.gtr.y", "P.react.z"]
PID = ["P.ngcer.npeSum", "P.hgcer.npeSum", "P.cal.etottracknorm"]
READ = FP + [RASTER] + RECO + PID


class Coupling(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, inverse=False):
        kept = x * self.mask
        log_s, t = self.net(kept).chunk(2, dim=1)
        log_s = .65 * torch.tanh(log_s) * (1 - self.mask); t = t * (1 - self.mask)
        return kept + (1 - self.mask) * ((x - t) * torch.exp(-log_s) if inverse else x * torch.exp(log_s) + t)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        masks = ([1,0,1,0,1], [0,1,0,1,0], [1,1,0,0,1], [0,0,1,1,0], [1,0,0,1,1], [0,1,1,0,0])
        self.layers = nn.ModuleList([Coupling(m) for m in masks])

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


def load_full_root() -> tuple[pd.DataFrame, dict[str, int]]:
    arr = uproot.open(RAW)["T"].arrays(READ, library="np")
    raw_n = len(arr[FP[0]])
    valid = np.ones(raw_n, dtype=bool)
    for name in READ: valid &= np.isfinite(arr[name])
    finite_n = int(valid.sum())
    valid &= np.logical_and.reduce([arr[name] > -9999 for name in FP])
    valid &= (arr["P.gtr.dp"] >= -25) & (arr["P.gtr.dp"] <= 22)
    valid &= (arr["P.gtr.th"] >= -.08) & (arr["P.gtr.th"] <= .08)
    valid &= (arr["P.gtr.ph"] >= -.06) & (arr["P.gtr.ph"] <= .06)
    valid &= (arr["P.react.z"] >= -120) & (arr["P.react.z"] <= 120)
    technical_n = int(valid.sum())
    # Physics-quality selection only; critically, no sieve coordinate or foil/hole condition.
    valid &= (arr["P.ngcer.npeSum"] >= 6) & (arr["P.hgcer.npeSum"] >= 0)
    valid &= (arr["P.cal.etottracknorm"] >= .8) & (arr["P.cal.etottracknorm"] <= 1.8)
    frame = pd.DataFrame({name: arr[name][valid] for name in READ})
    # Same continuous reconstruction used in the earlier flow experiment,
    # computed directly from raw ROOT branches without a sieve window cut.
    dp, th, ph = (frame[c].to_numpy() for c in ("P.gtr.dp", "P.gtr.th", "P.gtr.ph"))
    x, y = frame["P.gtr.x"].to_numpy(), frame["P.gtr.y"].to_numpy()
    frame["sieve_x"] = x + 253.0 * th
    frame["sieve_y"] = (-.019 * dp + .00019 * dp**2 + 213.0 * ph + y) + 40.0 * (-.00052 * dp + .0000052 * dp**2 + ph)
    return frame, {"root_entries": raw_n, "finite_reconstructed": finite_n, "technical_kinematic": technical_n, "technical_kinematic_pid_no_sieve_cut": int(len(frame))}


def contrast_colors(labels, centers):
    order = centers.sort_values(["sieve_y", "sieve_x"]).index.to_list()
    palette = [hsv_to_rgb((i / 24, .72, .88)) for i in range(24)]
    return {lab: palette[i % len(palette)] for i, lab in enumerate(order)}


def main():
    OUT.mkdir(parents=True, exist_ok=True); torch.manual_seed(SEED); np.random.seed(SEED)
    df, counts = load_full_root()
    optical = df[FP].to_numpy()
    raster = df[[RASTER]].to_numpy()
    condition = make_pipeline(SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), Ridge(alpha=2.0))
    condition.fit(raster, optical)
    corrected = np.column_stack([optical - condition.predict(raster), raster[:, 0]])
    x = RobustScaler(quantile_range=(5,95)).fit_transform(corrected).astype("float32")
    targets = df[["sieve_x", "sieve_y", "P.gtr.y"]].to_numpy()
    y = StandardScaler().fit_transform(targets).astype("float32")

    model = Flow(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=2048, shuffle=True)
    history = []
    for epoch in range(80):
        total = 0.0
        for xb, yb in loader:
            z = model(xb); loss = ((z[:, :3] - yb) ** 2).mean() + .02 * ((z[:, 3:] - xb[:, 3:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            total += float(loss.detach()) * len(xb)
        if epoch in (0, 19, 39, 79): history.append({"epoch": epoch + 1, "loss": total / len(x)})
    with torch.no_grad(): z = model(torch.from_numpy(x)).numpy()
    z[:, 3:] *= np.sqrt(.10)
    # Persist the learned Z-space coordinates.  They are needed for downstream
    # relative-geometry studies and are not used as any external sieve cut.
    latent_columns = [f"flow_z{i + 1}" for i in range(z.shape[1])]
    df.loc[:, latent_columns] = z
    torch.save({"state_dict": model.state_dict(), "latent_columns": latent_columns, "seed": SEED}, OUT / "raw_fullroot_flow_model.pt")
    # A conservative full-root setting: no target cluster count is supplied.
    clusterer = HDBSCAN(min_cluster_size=60, min_samples=10, cluster_selection_method="eom")
    labels = clusterer.fit_predict(z)
    df["flow_hdbscan_cluster"] = labels; df["hdbscan_probability"] = clusterer.probabilities_
    active = df.flow_hdbscan_cluster >= 0
    centers = df.loc[active].groupby("flow_hdbscan_cluster").agg(
        events=("flow_hdbscan_cluster", "size"), sieve_x=("sieve_x", "median"), sieve_y=("sieve_y", "median"),
        ytar=("P.gtr.y", "median"), probability=("hdbscan_probability", "median"))
    colors = contrast_colors(centers.index, centers)
    compact = df[[*FP, RASTER, *latent_columns, "sieve_x", "sieve_y", "P.gtr.y", "flow_hdbscan_cluster", "hdbscan_probability"]]
    compact.to_csv(OUT / "raw_fullroot_flow_hdbscan_labels.csv", index=False)
    centers.to_csv(OUT / "raw_fullroot_flow_hdbscan_centers.csv")
    summary = {**counts, "fit_inputs": [*FP, RASTER], "weak_continuous_targets": ["sieve_x", "sieve_y", "P.gtr.y"],
               "not_read": ["foil_position", "hole_id", "hole_row", "hole_col", "sieve_label", "cluster"],
               "selection": "finite + valid DC + kinematics + PID; explicitly no sieve-range, foil, or hole cut",
               "epochs": 80, "hdbscan": {"min_cluster_size": 60, "min_samples": 10, "selection": "eom",
                   "clusters": int(len(centers)), "noise_fraction": float((~active).mean()),
                   "median_cluster_events": float(centers.events.median()), "largest_cluster_fraction": float(centers.events.max() / len(df)),
                   "median_membership_probability": float(centers.probability.median())}, "loss": history}
    (OUT / "raw_fullroot_flow_hdbscan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sample = df.sample(min(100000, len(df)), random_state=SEED)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    sig = sample[sample.flow_hdbscan_cluster >= 0]; noise = sample[sample.flow_hdbscan_cluster < 0]
    point_colors = [colors[c] for c in sig.flow_hdbscan_cluster]
    for ax, xp, yp, title in ((axes[0,0], "sieve_x", "sieve_y", "All inferred clusters in reconstructed sieve plane"),
                              (axes[0,1], "sieve_x", "P.gtr.y", "All inferred clusters: sieve-x vs reconstructed ytar")):
        ax.scatter(noise[xp], noise[yp], c="0.75", s=.35, alpha=.18, linewidths=0, rasterized=True)
        ax.scatter(sig[xp], sig[yp], c=point_colors, s=.55, alpha=.60, linewidths=0, rasterized=True)
        ax.set(xlabel=xp, ylabel=yp, title=title); ax.grid(alpha=.15)
    axes[1,0].scatter(centers.sieve_x, centers.sieve_y, c=centers.ytar, cmap="coolwarm", s=np.clip(centers.events / 8, 12, 160), alpha=.86, edgecolors="black", linewidths=.25)
    axes[1,0].set(xlabel="median reconstructed sieve_x", ylabel="median reconstructed sieve_y", title="Inferred cluster centres (size = population; colour = median ytar)"); axes[1,0].grid(alpha=.15)
    axes[1,1].hist(centers.ytar, bins=35, color="#2E74B5", alpha=.85)
    axes[1,1].set(xlabel="inferred-cluster median reconstructed ytar", ylabel="number of inferred clusters", title="Unsupervised cluster-centre ytar structure"); axes[1,1].grid(axis="y", alpha=.15)
    fig.suptitle(f"Unskimmed ROOT run 25521: all {len(df):,} quality events, no sieve/foil/hole selection", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "raw_fullroot_flow_hdbscan_overview.png", dpi=220, bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
