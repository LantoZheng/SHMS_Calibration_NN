#!/usr/bin/env python3
"""Measure the existing sieve-plane clustering as an external reference only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FP = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
SIEVE = ["sieve_x", "sieve_y"]
REF = ["foil_position", "cluster"]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "SHMS_Calibration_NN" / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(path, usecols=FP + SIEVE + REF).dropna()
    rng = np.random.default_rng(25521)
    frame = frame.iloc[rng.choice(len(frame), 30000, replace=False)].reset_index(drop=True)
    fp = RobustScaler(quantile_range=(5, 95)).fit_transform(frame[FP])
    sieve = RobustScaler(quantile_range=(5, 95)).fit_transform(frame[SIEVE])
    per_foil = {}
    for foil in sorted(frame.foil_position.unique()):
        mask = frame.foil_position.to_numpy() == foil
        labels = frame.loc[mask, "cluster"].to_numpy()
        fp_foil, sieve_foil = fp[mask], sieve[mask]
        take = rng.choice(len(labels), min(5000, len(labels)), replace=False)
        local = {"n_events": int(len(labels)), "clusters": int(np.unique(labels).size), "silhouette": {}, "k20_same_cluster": {}}
        for name, data in {"fp": fp_foil, "sieve": sieve_foil}.items():
            local["silhouette"][name] = float(silhouette_score(data[take], labels[take]))
            neighbors = NearestNeighbors(n_neighbors=21, n_jobs=-1).fit(data).kneighbors(return_distance=False)[:, 1:]
            same = (labels[:, None] == labels[neighbors]).mean(axis=1)
            local["k20_same_cluster"][name] = {"mean": float(same.mean()), "median": float(np.median(same))}
        per_foil[str(int(foil))] = local
    summary = {
        "n_events": int(len(frame)),
        "reference": "existing per-foil sieve-plane HDBSCAN clusters; used only after the fact",
        "reference_cluster_count": int(sum(v["clusters"] for v in per_foil.values())),
        "per_foil": per_foil,
        "interpretation": "The FP numbers quantify how much information the existing sieve clusters leave locally visible in measured focal-plane coordinates.",
    }
    (out / "reference_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
