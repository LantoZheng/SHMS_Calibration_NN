#!/usr/bin/env python3
"""Stricter metric test: hold out entire sieve clusters, not random events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
ADC = ["P_rb_raster_fryaRawAdc", "P_rb_raster_frybRawAdc"]
TARGET = ["cluster_center_x", "cluster_center_y", "foil_ytar_center"]


def score(a, y, labels):
    neighbors = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(a).kneighbors(return_distance=False)[:, 1:]
    target_neighbors = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(y).kneighbors(return_distance=False)[:, 1:]
    overlap = np.mean([len(set(x) & set(z)) / 20 for x, z in zip(neighbors, target_neighbors)])
    purity = (labels[:, None] == labels[neighbors]).mean()
    sample = np.linspace(0, len(a) - 1, min(5000, len(a)), dtype=int)
    return {"target_knn_overlap_k20": float(overlap), "same_cluster_fraction_k20": float(purity), "silhouette": float(silhouette_score(a[sample], labels[sample]))}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    columns = FP + ADC + TARGET + ["foil_position", "cluster"]
    df = pd.read_csv(path, usecols=columns).dropna()
    rng = np.random.default_rng(25521)
    df = df.iloc[rng.choice(len(df), 60000, replace=False)].reset_index(drop=True)
    labels = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    holes = np.unique(labels)
    test_holes = rng.choice(holes, size=int(np.ceil(.20 * len(holes))), replace=False)
    test = np.isin(labels, test_holes)
    train = ~test
    df["fry_proxy"] = df[ADC].mean(axis=1)
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP + ["fry_proxy"]])
    y = StandardScaler().fit_transform(df[TARGET])
    linear = Ridge(alpha=1.0).fit(x[train], y[train])
    mlp = MLPRegressor(hidden_layer_sizes=(96, 64), activation="relu", alpha=2e-4, learning_rate_init=1e-3,
                       early_stopping=True, validation_fraction=.12, max_iter=300, random_state=25521).fit(x[train], y[train])
    linear_pred, mlp_pred = linear.predict(x[test]), mlp.predict(x[test])
    report = {
        "n_events": int(len(df)), "train_events": int(train.sum()), "test_events": int(test.sum()),
        "held_out_sieve_clusters": int(len(test_holes)), "input_features": FP + ["fry_proxy"], "weak_targets": TARGET,
        "split": "all events from held-out sieve clusters excluded from training",
        "metrics": {
            "raw_fp_fry": score(x[test], y[test], labels[test]),
            "linear_metric": score(linear_pred, y[test], labels[test]),
            "nonlinear_metric": score(mlp_pred, y[test], labels[test]),
        },
    }
    report["metrics"]["linear_metric"]["target_rmse_scaled"] = float(np.sqrt(mean_squared_error(y[test], linear_pred)))
    report["metrics"]["nonlinear_metric"]["target_rmse_scaled"] = float(np.sqrt(mean_squared_error(y[test], mlp_pred)))
    (out / "metric_flattening_holeholdout_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
