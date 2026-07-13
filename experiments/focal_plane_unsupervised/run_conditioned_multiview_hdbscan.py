#!/usr/bin/env python3
"""Hierarchical density clustering using FP plus a tunable sieve weak prior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
SIEVE = ["sieve_x", "sieve_y"]
READ = FP + SIEVE + ["P_gtr_y", "foil_position", "cluster"]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(data, usecols=READ).dropna()
    rng = np.random.default_rng(25521)
    df = df.iloc[rng.choice(len(df), 15000, replace=False)].reset_index(drop=True)
    gmm = GaussianMixture(n_components=3, random_state=25521).fit(df[["P_gtr_y"]])
    group = gmm.predict(df[["P_gtr_y"]])
    reference = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    fp = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP])
    sieve = RobustScaler(quantile_range=(5, 95)).fit_transform(df[SIEVE])
    rows = []
    for weight in (0.01, 0.03, 0.10, 0.30, 1.0):
        x = np.column_stack([fp, np.sqrt(weight) * sieve])
        for min_size in (15, 30, 60):
          for selection in ("leaf", "eom"):
            output = np.full(len(df), -1, dtype=int)
            offset = 0
            for g in range(3):
                mask = group == g
                labels = HDBSCAN(min_cluster_size=min_size, min_samples=max(5, min_size // 3), cluster_selection_method=selection).fit_predict(x[mask])
                active = labels >= 0
                n_local = int(labels[active].max() + 1) if active.any() else 0
                labels[active] += offset
                offset += n_local
                output[mask] = labels
            active = output >= 0
            sizes = np.bincount(output[active]) if active.any() else np.array([])
            rows.append({
                "prior_weight": weight, "min_cluster_size": min_size, "selection": selection,
                "clusters": int(len(sizes)), "noise_fraction": float(1 - active.mean()),
                "median_cluster_size": float(np.median(sizes)) if len(sizes) else 0.0,
                "max_cluster_fraction": float(sizes.max() / len(df)) if len(sizes) else 0.0,
                "reference_ami": float(adjusted_mutual_info_score(reference, output)),
                "reference_ari": float(adjusted_rand_score(reference, output)),
            })
    scan = pd.DataFrame(rows)
    scan.to_csv(out / "conditioned_multiview_hdbscan_scan.csv", index=False)
    report = {
        "n_events": int(len(df)), "features": FP,
        "weak_prior": "sqrt(lambda) * robust-scaled sieve_x/y appended to FP",
        "conditioning": "3-component GMM on reconstructed P_gtr_y",
        "reference_evaluation_only": "existing sieve-plane HDBSCAN cluster identity",
        "best_ami": scan.sort_values("reference_ami", ascending=False).head(1).to_dict(orient="records")[0],
        "best_ari": scan.sort_values("reference_ari", ascending=False).head(1).to_dict(orient="records")[0],
    }
    (out / "conditioned_multiview_hdbscan_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
