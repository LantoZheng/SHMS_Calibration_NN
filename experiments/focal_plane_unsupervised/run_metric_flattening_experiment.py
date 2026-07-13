#!/usr/bin/env python3
"""Learn a nonlinear FP+fry embedding toward sieve-plane plus ytar coordinates.

This is a held-out metric experiment.  The embedding is trained on the current
reconstructed sieve/foil coordinates as weak targets; existing hole IDs enter
only after training to measure local cluster behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
ADC = ["P_rb_raster_fryaRawAdc", "P_rb_raster_frybRawAdc"]
TARGET = ["cluster_center_x", "cluster_center_y", "foil_ytar_center"]
READ = FP + ADC + TARGET + ["foil_position", "cluster"]


def mean_knn_overlap(a: np.ndarray, reference: np.ndarray, k: int = 20) -> float:
    a_idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(a).kneighbors(return_distance=False)[:, 1:]
    r_idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(reference).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a_idx, r_idx)]))


def same_cluster_fraction(a: np.ndarray, labels: np.ndarray, k: int = 20) -> float:
    neighbors = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(a).kneighbors(return_distance=False)[:, 1:]
    return float((labels[:, None] == labels[neighbors]).mean())


def pair_distance_correlation(a: np.ndarray, target: np.ndarray, seed: int = 25521) -> float:
    rng = np.random.default_rng(seed)
    n = len(a)
    left, right = rng.integers(0, n, size=(2, 20000))
    da = np.linalg.norm(a[left] - a[right], axis=1)
    dt = np.linalg.norm(target[left] - target[right], axis=1)
    return float(np.corrcoef(da, dt)[0, 1])


def evaluate(name: str, embedding: np.ndarray, target: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    sample = np.linspace(0, len(embedding) - 1, min(5000, len(embedding)), dtype=int)
    return {
        "target_knn_overlap_k20": mean_knn_overlap(embedding, target),
        "same_sieve_cluster_fraction_k20": same_cluster_fraction(embedding, labels),
        "distance_correlation_to_target": pair_distance_correlation(embedding, target),
        "sieve_cluster_silhouette": float(silhouette_score(embedding[sample], labels[sample])),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(path, usecols=READ).dropna()
    rng = np.random.default_rng(25521)
    df = df.iloc[rng.choice(len(df), 60000, replace=False)].reset_index(drop=True)
    df["fry_proxy"] = df[ADC].mean(axis=1)
    x_raw = df[FP + ["fry_proxy"]].to_numpy(np.float64)
    y_raw = df[TARGET].to_numpy(np.float64)
    labels = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    x_scaler, y_scaler = RobustScaler(quantile_range=(5, 95)), StandardScaler()
    x = x_scaler.fit_transform(x_raw)
    y = y_scaler.fit_transform(y_raw)
    indices = np.arange(len(df))
    train, test = train_test_split(indices, test_size=.25, random_state=25521, stratify=labels)

    linear = Ridge(alpha=1.0).fit(x[train], y[train])
    mlp = MLPRegressor(hidden_layer_sizes=(96, 64), activation="relu", alpha=2e-4, learning_rate_init=1e-3,
                       early_stopping=True, validation_fraction=.12, max_iter=300, random_state=25521)
    mlp.fit(x[train], y[train])
    pred_linear, pred_mlp = linear.predict(x[test]), mlp.predict(x[test])
    y_test, labels_test = y[test], labels[test]
    methods = {
        "raw_fp_fry": x[test],
        "linear_metric": pred_linear,
        "nonlinear_metric": pred_mlp,
    }
    metrics = {name: evaluate(name, emb, y_test, labels_test) for name, emb in methods.items()}
    metrics["linear_metric"]["target_rmse_scaled"] = float(np.sqrt(mean_squared_error(y_test, pred_linear)))
    metrics["nonlinear_metric"]["target_rmse_scaled"] = float(np.sqrt(mean_squared_error(y_test, pred_mlp)))
    metrics["raw_fp_fry"]["target_rmse_scaled"] = None

    # Physical-coordinate prediction plots and metric comparisons.
    physical = y_scaler.inverse_transform(pred_mlp)
    truth = y_raw[test]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, col, name in zip(axes, range(3), TARGET):
        ax.hexbin(truth[:, col], physical[:, col], gridsize=65, bins="log", mincnt=1, cmap="viridis")
        lo, hi = np.quantile(truth[:, col], [.01, .99])
        ax.plot([lo, hi], [lo, hi], color="tomato", lw=1.5)
        ax.set(title=f"held-out prediction: {name}", xlabel="reference", ylabel="nonlinear embedding")
    fig.savefig(out / "07_metric_embedding_predictions.png", dpi=180)
    plt.close(fig)

    keys = ["target_knn_overlap_k20", "same_sieve_cluster_fraction_k20", "distance_correlation_to_target", "sieve_cluster_silhouette"]
    chart = pd.DataFrame(metrics).T[keys]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    chart.plot(kind="bar", ax=ax)
    ax.set(title="Held-out geometry: raw FP+fry vs learned metrics", ylabel="score", xlabel="embedding")
    ax.legend(loc="best")
    ax.axhline(0, color="black", lw=.8)
    fig.savefig(out / "08_metric_flattening_comparison.png", dpi=180)
    plt.close(fig)

    report = {
        "n_events": int(len(df)), "train_events": int(len(train)), "test_events": int(len(test)),
        "input_features": FP + ["fry_proxy"], "weak_targets": TARGET,
        "existing_cluster_use": "stratification and post-hoc held-out evaluation only",
        "models": {"linear": "Ridge", "nonlinear": "MLP 5->96->64->3"},
        "metrics": metrics,
        "interpretation": "A useful flattening metric should improve held-out target-neighborhood overlap and same-sieve-cluster locality without relying on hole identity as an input.",
    }
    (out / "metric_flattening_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
