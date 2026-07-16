"""Stage-2 conservative refinement for the FP5D flow + HDBSCAN pipeline.

Inputs are the stage-1 global FP5D density labels.  Reconstructed sieve
coordinates are used only to find cells containing more than one close cluster;
they are never used as branch labels.  In each cell, high-membership-probability
stage-1 members seed a shrinkage-LDA field in original FP5D.  The field routes
ambiguous members among the existing local branches and leaves everything else
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "raw_fullroot_flow_hdbscan_labels.csv"
FEATURES = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
SEED_PROBABILITY = 0.80
SIEVE_CONFLICT_RADIUS_CM = 0.90
MIN_SEED_EVENTS_PER_BRANCH = 20
MIN_COMPONENT_BRANCHES = 2
REASSIGN_PROBABILITY = 0.90
REASSIGN_MARGIN = 0.20
RANDOM_STATE = 25521


class UnionFind:
    def __init__(self, nodes: list[int]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def connected_conflict_cells(centers: pd.DataFrame) -> list[list[int]]:
    ids = centers.index.astype(int).to_list()
    xy = centers[["sieve_x", "sieve_y"]].to_numpy()
    tree = cKDTree(xy)
    uf = UnionFind(ids)
    for first, second in tree.query_pairs(SIEVE_CONFLICT_RADIUS_CM):
        uf.union(ids[first], ids[second])
    groups: dict[int, list[int]] = {}
    for cluster in ids:
        groups.setdefault(uf.find(cluster), []).append(cluster)
    return [sorted(group) for group in groups.values() if len(group) >= MIN_COMPONENT_BRANCHES]


def stable_colour(label: int) -> tuple[float, float, float, float]:
    return plt.get_cmap("hsv")((int(label) * 0.61803398875) % 1.0)


def main() -> None:
    raw = pd.read_csv(SOURCE)
    active = raw.flow_hdbscan_cluster >= 0
    data = raw.loc[active].copy()
    centers = data.groupby("flow_hdbscan_cluster").agg(
        events=("flow_hdbscan_cluster", "size"), sieve_x=("sieve_x", "median"),
        sieve_y=("sieve_y", "median"), median_probability=("hdbscan_probability", "median"),
    )
    cells = connected_conflict_cells(centers)
    scaler = RobustScaler(quantile_range=(5, 95))
    fp = pd.DataFrame(scaler.fit_transform(data[FEATURES]), index=data.index, columns=FEATURES)

    refined = raw.flow_hdbscan_cluster.copy()
    local_component = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    local_seed = pd.Series(False, index=raw.index)
    local_probability = pd.Series(np.nan, index=raw.index)
    local_margin = pd.Series(np.nan, index=raw.index)
    local_z3 = pd.Series(np.nan, index=raw.index)
    diagnostics: list[dict[str, object]] = []

    for component_id, clusters in enumerate(cells):
        part = data.loc[data.flow_hdbscan_cluster.isin(clusters)].copy()
        label_to_position = {cluster: position for position, cluster in enumerate(clusters)}
        y = part.flow_hdbscan_cluster.map(label_to_position).to_numpy()
        x = fp.loc[part.index].to_numpy()
        seeds = part.hdbscan_probability.to_numpy() >= SEED_PROBABILITY
        counts = np.bincount(y[seeds], minlength=len(clusters))
        # A cell with insufficient seeds is left untouched and recorded.
        if counts.min() < MIN_SEED_EVENTS_PER_BRANCH:
            diagnostics.append({
                "component": component_id, "clusters": ";".join(map(str, clusters)), "branches": len(clusters),
                "events": len(part), "seed_events": int(seeds.sum()), "status": "skipped_insufficient_seeds",
                "minimum_seed_count": int(counts.min()),
            })
            continue

        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = cross_val_predict(estimator, x[seeds], y[seeds], cv=cv, method="predict")
        oof_balanced_accuracy = float(balanced_accuracy_score(y[seeds], oof))
        estimator.fit(x[seeds], y[seeds])
        probabilities = estimator.predict_proba(x)
        order = np.argsort(probabilities, axis=1)
        best_position = order[:, -1]
        best_probability = probabilities[np.arange(len(part)), best_position]
        runner_up = probabilities[np.arange(len(part)), order[:, -2]]
        margin = best_probability - runner_up
        candidate = np.asarray(clusters, dtype=int)[best_position]
        # Seeds are immutable.  Only non-seeds passing both confidence guards move.
        can_move = (~seeds) & (best_probability >= REASSIGN_PROBABILITY) & (margin >= REASSIGN_MARGIN)
        original = part.flow_hdbscan_cluster.to_numpy(dtype=int)
        revised = original.copy()
        revised[can_move] = candidate[can_move]

        local_component.loc[part.index] = component_id
        local_seed.loc[part.index] = seeds
        local_probability.loc[part.index] = best_probability
        local_margin.loc[part.index] = margin
        if len(clusters) == 2:
            score = estimator.decision_function(x)
            # Display sign: increasing z3 follows increasing local median reconstructed ytar.
            if part.loc[y == 1, "P.gtr.y"].median() < part.loc[y == 0, "P.gtr.y"].median():
                score *= -1
            local_z3.loc[part.index] = score
        refined.loc[part.index] = revised
        pair_distance = centers.loc[clusters, ["sieve_x", "sieve_y"]].to_numpy()
        diagnostics.append({
            "component": component_id, "clusters": ";".join(map(str, clusters)), "branches": len(clusters),
            "events": len(part), "seed_events": int(seeds.sum()), "status": "resolved",
            "minimum_seed_count": int(counts.min()), "oof_seed_balanced_accuracy": oof_balanced_accuracy,
            "reassigned_events": int((revised != original).sum()),
            "mean_router_probability": float(best_probability.mean()), "mean_router_margin": float(margin.mean()),
            "maximum_sieve_centre_span_cm": float(np.max(np.linalg.norm(pair_distance[:, None] - pair_distance[None, :], axis=2))),
        })

    output = raw.copy()
    output["linear_local_field_cluster"] = refined.astype(int)
    output["local_field_component"] = local_component
    output["local_field_seed"] = local_seed
    output["local_field_probability"] = local_probability
    output["local_field_margin"] = local_margin
    output["local_field_z3_binary_cells"] = local_z3
    output["local_field_changed"] = (output.linear_local_field_cluster != output.flow_hdbscan_cluster) & active
    component_df = pd.DataFrame(diagnostics)
    output.to_csv(RESULTS / "linear_local_field_pipeline_labels.csv", index=False)
    component_df.to_csv(RESULTS / "linear_local_field_pipeline_components.csv", index=False)

    resolved = component_df.loc[component_df.status == "resolved"].copy()
    summary = {
        "input": str(SOURCE.name), "stage_1": "continuous-prior FP5D flow + HDBSCAN",
        "stage_2": "sieve-local, high-confidence-seeded shrinkage-LDA field in original FP5D",
        "sieve_used_for": "finding conflict cells only; never as a label or a clustering coordinate",
        "ytar_used_for": "binary z3 display orientation only; never as a routing label",
        "parameters": {"sieve_conflict_radius_cm": SIEVE_CONFLICT_RADIUS_CM, "seed_probability": SEED_PROBABILITY,
                       "min_seed_events_per_branch": MIN_SEED_EVENTS_PER_BRANCH, "reassign_probability": REASSIGN_PROBABILITY,
                       "reassign_margin": REASSIGN_MARGIN},
        "clusters_in_stage_1": int(len(centers)), "candidate_conflict_cells": int(len(cells)),
        "resolved_cells": int((component_df.status == "resolved").sum()),
        "skipped_cells": int((component_df.status != "resolved").sum()),
        "events_routed_by_local_fields": int(output.local_field_component.notna().sum()),
        "conservative_reassignments": int(output.local_field_changed.sum()),
        "median_oof_seed_balanced_accuracy": float(resolved.oof_seed_balanced_accuracy.median()) if len(resolved) else None,
    }
    (RESULTS / "linear_local_field_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sample = output.sample(min(90000, len(output)), random_state=RANDOM_STATE)
    changed = output.loc[output.local_field_changed]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for ax, column, title in [
        (axes[0, 0], "flow_hdbscan_cluster", "Stage 1: global FP5D flow + HDBSCAN"),
        (axes[0, 1], "linear_local_field_cluster", "Stage 2: conservative local-field routing"),
    ]:
        sig = sample.loc[sample[column] >= 0]
        ax.scatter(sample.loc[sample[column] < 0, "sieve_x"], sample.loc[sample[column] < 0, "sieve_y"], c="0.75", s=.35, alpha=.12, linewidths=0)
        ax.scatter(sig.sieve_x, sig.sieve_y, c=[stable_colour(label) for label in sig[column]], s=.55, alpha=.62, linewidths=0)
        ax.set(xlabel=r"reconstructed $x_{sieve}$", ylabel=r"reconstructed $y_{sieve}$", title=title); ax.grid(alpha=.15)
    axes[1, 0].scatter(sample.sieve_x, sample.sieve_y, c="0.78", s=.35, alpha=.15, linewidths=0)
    axes[1, 0].scatter(changed.sieve_x, changed.sieve_y, c="crimson", s=11, alpha=.90, linewidths=0)
    axes[1, 0].set(xlabel=r"reconstructed $x_{sieve}$", ylabel=r"reconstructed $y_{sieve}$", title=f"Events conservatively rerouted by local fields: {len(changed):,}"); axes[1, 0].grid(alpha=.15)
    if len(resolved):
        axes[1, 1].scatter(resolved.branches, resolved.oof_seed_balanced_accuracy, s=np.clip(resolved.events / 8, 20, 180), alpha=.78, c=resolved.maximum_sieve_centre_span_cm, cmap="viridis")
        axes[1, 1].axhline(.90, color="crimson", lw=1, alpha=.65)
    axes[1, 1].set(xlabel="branches in local conflict cell", ylabel="5-fold OOF seed balanced accuracy", ylim=(0, 1.04), title="Audit of local-field cells (size = events; colour = sieve span)"); axes[1, 1].grid(alpha=.15)
    fig.suptitle("Two-stage FP5D clustering with conservative linear local-coordinate fields", fontsize=16, fontweight="bold")
    fig.savefig(RESULTS / "linear_local_field_pipeline_diagnostics.png", dpi=200)


if __name__ == "__main__":
    main()
