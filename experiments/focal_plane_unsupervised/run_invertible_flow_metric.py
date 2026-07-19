"""Weakly supervised, invertible 5D transport for focal-plane geometry.

The map is RealNVP 5D -> 5D.  Its first three coordinates are softly trained
against the existing sieve/foil reconstruction; two residual coordinates remain
in the metric and the map is exactly invertible.  Complete reference holes are
held out during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
TARGET = ["cluster_center_x", "cluster_center_y", "foil_ytar_center"]
SEED = 25521


class Coupling(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, inverse=False):
        masked = x * self.mask
        scale, shift = self.net(masked).chunk(2, dim=1)
        scale = 0.65 * torch.tanh(scale) * (1 - self.mask)
        shift = shift * (1 - self.mask)
        if inverse:
            return masked + (1 - self.mask) * ((x - shift) * torch.exp(-scale))
        return masked + (1 - self.mask) * (x * torch.exp(scale) + shift)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Coupling([1, 0, 1, 0, 1]), Coupling([0, 1, 0, 1, 0]),
            Coupling([1, 1, 0, 0, 1]), Coupling([0, 0, 1, 1, 0]),
            Coupling([1, 0, 0, 1, 1]), Coupling([0, 1, 1, 0, 0]),
        ])

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

    def inverse(self, z):
        for layer in reversed(self.layers): z = layer(z, inverse=True)
        return z


def geometry_score(a, y, labels):
    nn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(a).kneighbors(return_distance=False)[:, 1:]
    ynn = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(y).kneighbors(return_distance=False)[:, 1:]
    overlap = np.mean([len(set(p) & set(q)) / 20 for p, q in zip(nn, ynn)])
    sample = np.linspace(0, len(a)-1, min(5000, len(a)), dtype=int)
    return {
        "target_knn_overlap_k20": float(overlap),
        "same_reference_cluster_fraction_k20": float((labels[:, None] == labels[nn]).mean()),
        "reference_cluster_silhouette": float(silhouette_score(a[sample], labels[sample])),
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    root = Path(__file__).resolve().parents[2]
    path = root / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).parent / "results"; out.mkdir(exist_ok=True)
    columns = FEATURES + TARGET + ["foil_position", "cluster"]
    df = pd.read_csv(path, usecols=columns).dropna()
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.choice(len(df), 60000, replace=False)].reset_index(drop=True)
    labels = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    holes = np.unique(labels); test_holes = rng.choice(holes, size=int(np.ceil(.20 * len(holes))), replace=False)
    test = np.isin(labels, test_holes); train = ~test
    xs = RobustScaler(quantile_range=(5,95)).fit(df.loc[train, FEATURES])
    ys = StandardScaler().fit(df.loc[train, TARGET])
    x, y = xs.transform(df[FEATURES]).astype("float32"), ys.transform(df[TARGET]).astype("float32")

    model = Flow(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])), batch_size=512, shuffle=True)
    history = []
    for epoch in range(100):
        model.train(); total = 0.0
        for xb, yb in loader:
            z = model(xb)
            # The two residual axes stay present and near their input scale;
            # they are not discarded from the learned 5D metric.
            loss = ((z[:, :3] - yb)**2).mean() + 0.02*((z[:, 3:] - xb[:, 3:])**2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach()) * len(xb)
        if epoch in (0, 24, 49, 99): history.append({"epoch": epoch+1, "loss": total/train.sum()})
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(x)).numpy()
        invert_err = torch.max(torch.abs(model.inverse(model(torch.from_numpy(x[:1024]))) - torch.from_numpy(x[:1024]))).item()
    raw = geometry_score(x[test], y[test], labels[test])
    flow_equal = geometry_score(z[test], y[test], labels[test])
    # The residual axes retain nonzero weight; this is a 5D metric, not a 3D reduction.
    weighted = z.copy(); weighted[:, 3:] *= np.sqrt(0.10)
    flow_weighted = geometry_score(weighted[test], y[test], labels[test])
    flow_equal["target_rmse_scaled"] = float(np.sqrt(mean_squared_error(y[test], z[test, :3])))
    flow_weighted["target_rmse_scaled"] = flow_equal["target_rmse_scaled"]
    cluster_rows = []
    for space_name, space in (("raw_5d", x[test]), ("flow_5d_equal", z[test]), ("flow_5d_residual_weight_0p10", weighted[test])):
        for min_size in (15, 30, 60):
            for selection in ("eom", "leaf"):
                predicted = HDBSCAN(min_cluster_size=min_size, min_samples=max(5, min_size // 3), cluster_selection_method=selection).fit_predict(space)
                active = predicted >= 0
                sizes = np.bincount(predicted[active]) if active.any() else np.array([])
                cluster_rows.append({
                    "space": space_name, "min_cluster_size": min_size, "selection": selection,
                    "clusters": int(len(sizes)), "noise_fraction": float(1-active.mean()),
                    "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
                    "max_cluster_fraction": float(sizes.max()/len(space)) if len(sizes) else 0.0,
                    "reference_hole_ami": float(adjusted_mutual_info_score(labels[test], predicted)),
                    "reference_hole_ari": float(adjusted_rand_score(labels[test], predicted)),
                })
    cluster_scan = pd.DataFrame(cluster_rows)
    cluster_scan.to_csv(out / "invertible_flow_hdbscan_holeholdout_scan.csv", index=False)
    usable = cluster_scan[(cluster_scan.clusters >= 20) & (cluster_scan.noise_fraction < .50) & (cluster_scan.max_cluster_fraction < .10)]
    report = {
        "n_events": int(len(df)), "train_events": int(train.sum()), "test_events": int(test.sum()),
        "held_out_complete_reference_holes": int(len(test_holes)), "input_features": FEATURES, "weak_targets": TARGET,
        "split": "all events from selected foil+sieve reference clusters excluded from training",
        "model": "6-coupling RealNVP, exactly invertible 5D-to-5D; first 3 axes weakly supervised, final 2 residual axes retained",
        "training_checkpoints": history, "round_trip_max_abs_error": float(invert_err),
        "metrics": {"raw_5d": raw, "flow_full_5d_equal_weight": flow_equal, "flow_full_5d_residual_weight_0p10": flow_weighted},
        "hdbscan_on_held_out_holes": {
            "selection_rule": "clusters>=20, noise<0.50, largest cluster<10%; then maximal reference-hole AMI",
            "best_candidate": usable.sort_values(["reference_hole_ami", "reference_hole_ari"], ascending=False).head(1).to_dict(orient="records")[0] if not usable.empty else None,
            "raw_baseline_best": cluster_scan[cluster_scan.space=="raw_5d"].sort_values("reference_hole_ami", ascending=False).head(1).to_dict(orient="records")[0],
        },
        "caution": "Targets are reconstructed sieve/foil references from this run; held-out holes test interpolation but not external physical truth or cross-run transfer."
    }
    (out / "invertible_flow_metric_holeholdout_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
