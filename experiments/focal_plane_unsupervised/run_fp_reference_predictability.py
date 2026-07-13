#!/usr/bin/env python3
"""Post-hoc FP predictability diagnostic for the existing sieve clusters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
REF = ["foil_position", "cluster"]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(path, usecols=FP + REF).dropna()
    rng = np.random.default_rng(25521)
    df = df.iloc[rng.choice(len(df), 30000, replace=False)].reset_index(drop=True)
    x = RobustScaler(quantile_range=(5, 95)).fit_transform(df[FP])
    y = (df.foil_position.astype(int) * 1000 + df.cluster.astype(int)).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=25521, stratify=y)
    model = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1).fit(x_train, y_train)
    probability = model.predict_proba(x_test)
    classes = model.classes_
    report = {
        "n_events": int(len(df)),
        "features": FP,
        "target_used_only_for_post_hoc_diagnostic": "existing sieve-plane HDBSCAN cluster identity",
        "random_event_holdout_note": "This is a local-overlap ceiling, not a cross-run generalization claim.",
        "top1_accuracy": float(accuracy_score(y_test, model.predict(x_test))),
        "top3_accuracy": float(top_k_accuracy_score(y_test, probability, k=3, labels=classes)),
        "top5_accuracy": float(top_k_accuracy_score(y_test, probability, k=5, labels=classes)),
    }
    (out / "fp_reference_predictability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
