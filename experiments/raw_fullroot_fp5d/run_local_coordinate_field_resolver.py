"""Resolve selected overlapping sieve neighbourhoods with a local FP5D field.

The output is deliberately a *local* branch coordinate, not a global foil
label.  A high-confidence subset of the existing density clusters seeds a
shrinkage-LDA normal.  The normal defines z3(x_sieve,y_sieve; FP5D) only inside
that conflict cell and reassigns remaining events in the two branches.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FEATURES = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
PAIRS = [("S1", 219, 220, 0), ("S2", 15, 68, 1), ("S3", 2, 3, 2), ("S4", 45, 56, 2)]
SEED_PROBABILITY = 0.80


def auc_abs(score: np.ndarray, target: np.ndarray) -> float:
    auc = roc_auc_score(target, score)
    return float(max(auc, 1.0 - auc))


def main() -> None:
    source = RESULTS / "raw_fullroot_flow_hdbscan_labels.csv"
    raw = pd.read_csv(source)
    data = raw.loc[raw.flow_hdbscan_cluster >= 0].copy()
    fp = RobustScaler(quantile_range=(5, 95)).fit_transform(data[FEATURES])
    fp = pd.DataFrame(fp, index=data.index, columns=FEATURES)
    resolved_parts, summary = [], []
    fig, axes = plt.subplots(3, 4, figsize=(21, 12), constrained_layout=True)
    colours = np.array(["tab:blue", "tab:orange"])

    for col, (tag, first, second, previous_band) in enumerate(PAIRS):
        part = data.loc[data.flow_hdbscan_cluster.isin([first, second])].copy()
        target = (part.flow_hdbscan_cluster.to_numpy() == second).astype(int)
        x = fp.loc[part.index].to_numpy()
        seeds = part.hdbscan_probability.to_numpy() >= SEED_PROBABILITY
        # Fall back safely when a small cluster contains too few high-confidence events.
        if min(np.bincount(target[seeds], minlength=2)) < 20:
            seeds[:] = True
        seed_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x[seeds], target[seeds])
        local_z = seed_model.decision_function(x)

        # z sign only supplies display/depth ordering; branch identity does not depend on it.
        if part.loc[target == 1, "P.gtr.y"].mean() < part.loc[target == 0, "P.gtr.y"].mean():
            local_z *= -1
            predicted = (local_z < 0).astype(int)
        else:
            predicted = (local_z >= 0).astype(int)
        # OOF seed AUC is the diagnostic; final field uses all eligible seed points.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=25521)
        oof = cross_val_predict(
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            x[seeds], target[seeds], cv=cv, method="decision_function",
        )
        oof_auc = auc_abs(oof, target[seeds])
        agreement = float((predicted == target).mean())
        part["local_field_pair"] = tag
        part["local_field_z3"] = local_z
        part["local_field_seed"] = seeds
        part["local_field_branch"] = predicted
        part["local_field_agrees_with_hdbscan"] = predicted == target
        resolved_parts.append(part)

        centers = part.groupby("flow_hdbscan_cluster")[["sieve_x", "sieve_y"]].mean()
        distance = float(np.linalg.norm(centers.loc[first] - centers.loc[second]))
        for branch, label in enumerate((first, second)):
            mask = predicted == branch
            seed_mask = mask & seeds
            nonseed_mask = mask & ~seeds
            axes[0, col].scatter(part.loc[nonseed_mask, "sieve_x"], part.loc[nonseed_mask, "sieve_y"], s=12, alpha=.48, color=colours[branch], linewidths=0)
            axes[0, col].scatter(part.loc[seed_mask, "sieve_x"], part.loc[seed_mask, "sieve_y"], s=13, alpha=.84, color=colours[branch], label=f"branch {label}", linewidths=0)
            axes[1, col].scatter(part.loc[nonseed_mask, "sieve_x"], local_z[nonseed_mask], s=12, alpha=.48, color=colours[branch], linewidths=0)
            axes[1, col].scatter(part.loc[seed_mask, "sieve_x"], local_z[seed_mask], s=13, alpha=.84, color=colours[branch], linewidths=0)
            bins = np.linspace(local_z.min(), local_z.max(), 30)
            axes[2, col].hist(local_z[mask], bins=bins, histtype="stepfilled", alpha=.35, color=colours[branch])
        for label in (first, second):
            axes[0, col].scatter(centers.loc[label, "sieve_x"], centers.loc[label, "sieve_y"], marker="*", s=78, color="red", edgecolor="white", linewidth=.4, zorder=5)
        axes[0, col].set(title=f"{tag}: local-field branch assignment\ncentre distance={distance:.3f} cm", xlabel=r"reconstructed $x_{sieve}$")
        axes[0, col].grid(alpha=.16); axes[0, col].legend(frameon=False, fontsize=8)
        axes[1, col].axhline(0, color="black", lw=.8, alpha=.5)
        axes[1, col].set(title=f"OOF seed AUC={oof_auc:.3f}; agreement={agreement:.3f}", xlabel=r"reconstructed $x_{sieve}$", ylabel=None)
        axes[1, col].grid(alpha=.16)
        axes[2, col].axvline(0, color="black", lw=.8, alpha=.5)
        axes[2, col].set(title="local $z_3$ branch distributions", xlabel=r"local FP5D $z_3$")
        axes[2, col].grid(alpha=.16)
        if col == 0:
            axes[0, col].set_ylabel(r"reconstructed $y_{sieve}$")
            axes[1, col].set_ylabel(r"local FP5D $z_3$")
            axes[2, col].set_ylabel("events / bin")
        summary.append({
            "pair": tag, "previous_ytar_band": previous_band, "clusters": [first, second],
            "events": int(len(part)), "high_confidence_seed_events": int(seeds.sum()),
            "out_of_fold_seed_auc": oof_auc, "all_event_agreement_with_initial_hdbscan": agreement,
            "reassigned_relative_to_initial_hdbscan": int((predicted != target).sum()),
            "sieve_center_distance_cm": distance,
            "field_definition": "z3_local = Fisher/LDA decision_function(robust_scaled FP5D), fitted on high-probability branch seeds",
        })

    fig.suptitle("Local FP5D coordinate field resolves branches in overlapping sieve neighbourhoods", fontsize=17, fontweight="bold")
    fig.savefig(RESULTS / "local_coordinate_field_resolved_neighbourhoods.png", dpi=200)
    pd.concat(resolved_parts, ignore_index=True).to_csv(RESULTS / "local_coordinate_field_resolved_events.csv", index=False)
    (RESULTS / "local_coordinate_field_resolution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
