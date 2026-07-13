#!/usr/bin/env python3
"""Cross-fitted FP consensus scores for existing sieve-plane cluster anchors.

This is intentionally not a replacement clusterer.  It asks which events have
their sieve-HDBSCAN assignment independently supported by focal-plane geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
READ = FP + ["foil_position", "cluster", "sieve_x", "sieve_y"]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    data = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(data, usecols=READ).dropna()
    rng = np.random.default_rng(25521)
    df = df.iloc[rng.choice(len(df), 30000, replace=False)].reset_index(drop=True)
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP])
    anchor = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    rank = np.zeros(len(df), dtype=np.int16)
    probability = np.zeros(len(df), dtype=np.float32)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=25521)
    for train, test in split.split(x, anchor):
        model = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1).fit(x[train], anchor[train])
        p = model.predict_proba(x[test])
        classes = model.classes_
        order = np.argsort(-p, axis=1)
        class_index = {int(c): i for i, c in enumerate(classes)}
        target_index = np.array([class_index[int(c)] for c in anchor[test]])
        rank[test] = (order == target_index[:, None]).argmax(axis=1) + 1
        probability[test] = p[np.arange(len(test)), target_index]
    result = df[["foil_position", "cluster", "sieve_x", "sieve_y"]].copy()
    result["fp_anchor_rank_oof"] = rank
    result["fp_anchor_probability_oof"] = probability
    result.to_csv(out / "sieve_anchor_fp_consensus_oof.csv", index=False)
    summary = {
        "n_events": int(len(df)),
        "anchor": "existing sieve-plane HDBSCAN cluster; used to train FP cross-check only",
        "fp_features": FP,
        "cross_fitting": "5-fold stratified out-of-fold KNN, k=25, distance weighted",
        "coverage": {f"rank_at_most_{k}": float((rank <= k).mean()) for k in (1, 2, 3, 5)},
        "mean_anchor_probability": float(probability.mean()),
        "high_confidence_consensus": {
            "definition": "anchor ranked first by out-of-fold FP model and probability >= 0.50",
            "coverage": float(((rank == 1) & (probability >= .50)).mean()),
        },
        "use": "retain high-confidence events as strong weak-label anchors; keep rank 2-3 as soft labels; flag the rest for exclusion or manual diagnostics.",
    }
    (out / "sieve_anchor_fp_consensus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
