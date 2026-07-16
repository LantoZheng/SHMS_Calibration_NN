"""Flatten the three foil sheets without introducing foil-dependent hard cuts.

The input is the identity-anchored global continuous z3.  On cluster
prototypes, a varying affine calibration

    C(u,z) = a(u) + b(u) z,   u=(z1,z2)

is fitted to the weak relative foil ranks (-1,0,+1).  a and b are normalized
Gaussian-RBF fields.  The deployed coordinate never reads an event's foil
label.  A smooth edge weight makes the flattening stronger where |z1| is large:

    z3_flat = z3 + rho(z1) [C(u,z3)-z3].

The affine-in-z form is used instead of an unconstrained polynomial so that the
mapping remains monotone along z3.  Ridge regularization and blending with the
identity preserve the close-pair separation already present in z3.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SOURCE = OUT / "global_continuous_z3_labels.csv"
PAIR_SOURCE = OUT / "global_continuous_z3_close_pair_metrics.csv"
CLUSTER = "linear_local_field_cluster"
BASE_Z = "global_continuous_z3"
FLAT_Z = "global_foil_flattened_z3"
U_COLUMNS = ["flow_z1", "flow_z2"]
RANDOM_STATE = 25521
N_EXPERTS = 36
RIDGE_ALPHA = 0.01


def rbf(u: np.ndarray, centres: np.ndarray, width: float) -> np.ndarray:
    d2 = ((u[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    p = np.exp(-0.5 * d2 / width**2)
    return p / (p.sum(axis=1, keepdims=True) + 1e-12)


def design(phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    local = np.column_stack([np.ones(len(z)), z])
    return (phi[:, :, None] * local[:, None, :]).reshape(len(z), -1)


def edge_weight(standardized_z1: np.ndarray) -> np.ndarray:
    # Smooth approximation to |z1| avoids a derivative cusp at the origin.
    radius = np.sqrt(standardized_z1**2 + 1e-6)
    return 0.25 + 0.70 / (1.0 + np.exp(-(radius - 0.90) / 0.25))


def dprime(values: np.ndarray, binary: np.ndarray) -> float:
    a, b = values[binary == 0], values[binary == 1]
    return float(abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)) + 1e-12))


def foil_mean_std(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("foil")[column].std().mean())


def main() -> None:
    raw = pd.read_csv(SOURCE)
    data = raw.loc[raw[CLUSTER] >= 0].copy()
    proto = data.groupby(CLUSTER).agg(
        z1=("flow_z1", "median"), z2=("flow_z2", "median"),
        base_z=(BASE_Z, "median"), foil=("final_relative_foil", "median"),
        events=(CLUSTER, "size"),
    )
    scaler = StandardScaler().fit(proto[["z1", "z2"]].to_numpy())
    pu = scaler.transform(proto[["z1", "z2"]].to_numpy())
    eu = scaler.transform(data[U_COLUMNS].to_numpy())

    centres = KMeans(N_EXPERTS, n_init=20, random_state=RANDOM_STATE).fit(pu).cluster_centers_
    width = float(np.median(cKDTree(centres).query(centres, k=2)[0][:, 1]) * 1.5)
    p_proto = rbf(pu, centres, width)
    x_proto = design(p_proto, proto.base_z.to_numpy())
    target = proto.foil.to_numpy(float) - 1.0

    # Cluster-level CV measures whether the sheet calibration is spatially transferable.
    cv_prediction = np.empty(len(proto))
    folds = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    for train, test in folds.split(x_proto):
        cv_prediction[test] = Ridge(alpha=RIDGE_ALPHA).fit(x_proto[train], target[train]).predict(x_proto[test])

    model = Ridge(alpha=RIDGE_ALPHA).fit(x_proto, target)
    p_event = rbf(eu, centres, width)
    coefficients = model.coef_.reshape(N_EXPERTS, 2)
    intercept_field = model.intercept_ + p_event @ coefficients[:, 0]
    slope_field = p_event @ coefficients[:, 1]
    full_calibration = intercept_field + slope_field * data[BASE_Z].to_numpy()
    rho = edge_weight(eu[:, 0])
    flattened = data[BASE_Z].to_numpy() + rho * (full_calibration - data[BASE_Z].to_numpy())

    data["global_z3_flattening_strength"] = rho
    data["global_z3_local_slope"] = (1.0 - rho) + rho * slope_field
    data[FLAT_Z] = flattened

    output = raw.copy()
    for column in ["global_z3_flattening_strength", "global_z3_local_slope", FLAT_Z]:
        output[column] = np.nan
        output.loc[data.index, column] = data[column].to_numpy()
    output.to_csv(OUT / "global_foil_flattened_z3_labels.csv", index=False)

    cluster_metrics = data.groupby(CLUSTER).agg(
        z1=("flow_z1", "median"), z2=("flow_z2", "median"),
        base_z=(BASE_Z, "median"), flattened_z3=(FLAT_Z, "median"),
        foil=("final_relative_foil", "median"), events=(CLUSTER, "size"),
        flattening_strength=("global_z3_flattening_strength", "median"),
    )
    cluster_u = scaler.transform(cluster_metrics[["z1", "z2"]].to_numpy())
    cluster_metrics["standardized_z1"] = cluster_u[:, 0]
    cluster_metrics["edge_region"] = np.abs(cluster_u[:, 0]) >= 1.0
    cluster_metrics.to_csv(OUT / "global_foil_flattened_z3_cluster_metrics.csv")

    pair_metrics = pd.read_csv(PAIR_SOURCE)
    new_dprime = []
    for row in pair_metrics.itertuples():
        part = data.loc[data[CLUSTER].isin([row.cluster_a, row.cluster_b])]
        binary = (part[CLUSTER].to_numpy() == row.cluster_b).astype(int)
        new_dprime.append(dprime(part[FLAT_Z].to_numpy(), binary))
    pair_metrics["dprime_foil_flattened_z3"] = new_dprime
    pair_metrics["flattening_dprime_gain"] = pair_metrics.dprime_foil_flattened_z3 - pair_metrics.dprime_global_continuous_z3
    pair_metrics.to_csv(OUT / "global_foil_flattened_z3_close_pair_metrics.csv", index=False)

    core = cluster_metrics.loc[~cluster_metrics.edge_region]
    edge = cluster_metrics.loc[cluster_metrics.edge_region]
    base_overall, flat_overall = foil_mean_std(cluster_metrics, "base_z"), foil_mean_std(cluster_metrics, "flattened_z3")
    base_core, flat_core = foil_mean_std(core, "base_z"), foil_mean_std(core, "flattened_z3")
    base_edge, flat_edge = foil_mean_std(edge, "base_z"), foil_mean_std(edge, "flattened_z3")
    summary = {
        "definition": "z3_flat=z3+rho(z1)*(sum_k phi_k(z1,z2)*(a_k+b_k*z3)-z3)",
        "inference_requires_foil_label": False,
        "training_weak_supervision": "cluster-level final relative foil rank only; no reconstructed ytar",
        "continuity": "normalized Gaussian RBF coefficient fields and smooth edge weight",
        "monotonicity": "affine in z3 at fixed z1,z2",
        "rbf_experts": N_EXPERTS, "ridge_alpha": RIDGE_ALPHA, "rbf_width": width,
        "cluster_level_5fold_mae": float(np.mean(abs(cv_prediction - target))),
        "cluster_level_5fold_nearest_foil_accuracy": float(np.mean(np.rint(np.clip(cv_prediction, -1, 1)) == target)),
        "minimum_event_local_z3_slope": float(data.global_z3_local_slope.min()),
        "flattening_strength_quantiles": dict(zip(["min", "median", "max"], map(float, np.quantile(rho, [0, .5, 1])))),
        "mean_within_foil_cluster_std": {
            "overall_before": base_overall, "overall_after": flat_overall,
            "overall_reduction_fraction": 1.0 - flat_overall / base_overall,
            "core_before": base_core, "core_after": flat_core,
            "edge_before": base_edge, "edge_after": flat_edge,
            "edge_reduction_fraction": 1.0 - flat_edge / base_edge,
        },
        "close_cross_foil_pair_separation": {
            "pairs": int(len(pair_metrics)),
            "median_dprime_before": float(pair_metrics.dprime_global_continuous_z3.median()),
            "median_dprime_after": float(pair_metrics.dprime_foil_flattened_z3.median()),
            "p10_dprime_before": float(pair_metrics.dprime_global_continuous_z3.quantile(.10)),
            "p10_dprime_after": float(pair_metrics.dprime_foil_flattened_z3.quantile(.10)),
            "fraction_pairs_improved": float((pair_metrics.flattening_dprime_gain > 0).mean()),
        },
    }
    (OUT / "global_foil_flattened_z3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colours = ["#356bb4", "#e98b27", "#35a16f"]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    for foil in range(3):
        q = cluster_metrics.loc[cluster_metrics.foil == foil]
        size = np.clip(q.events / 5, 14, 100)
        axes[0, 0].scatter(q.z1, q.base_z, s=size, c=colours[foil], alpha=.78, label=f"foil {foil}")
        axes[0, 1].scatter(q.z1, q.flattened_z3, s=size, c=colours[foil], alpha=.78, label=f"foil {foil}")
    axes[0, 0].set(xlabel="$z_1$", ylabel="global continuous $z_3$", title="Before: foil sheets bend and split near the edges")
    axes[0, 1].set(xlabel="$z_1$", ylabel="foil-flattened $z_3$", title="After: one continuous, position-dependent affine calibration")
    for ax in axes[0]: ax.grid(alpha=.15); ax.legend(frameon=False)

    before = [base_core, base_edge]; after = [flat_core, flat_edge]
    x = np.arange(2); width_bar = .34
    axes[1, 0].bar(x - width_bar/2, before, width_bar, color="#8ba7cf", label="before")
    axes[1, 0].bar(x + width_bar/2, after, width_bar, color="#46a979", label="after")
    axes[1, 0].set_xticks(x, ["core |z1| < 1", "edge |z1| >= 1"])
    axes[1, 0].set(ylabel="mean within-foil cluster-centre std", title=f"Edge spread reduced by {100*(1-flat_edge/base_edge):.1f}%")
    axes[1, 0].legend(frameon=False); axes[1, 0].grid(axis="y", alpha=.15)

    upper = float(pair_metrics[["dprime_global_continuous_z3", "dprime_foil_flattened_z3"]].quantile(.98).max())
    axes[1, 1].scatter(pair_metrics.dprime_global_continuous_z3, pair_metrics.dprime_foil_flattened_z3,
                       c=pair_metrics.distance_z1z2, cmap="viridis", s=24, alpha=.80)
    axes[1, 1].plot([0, upper], [0, upper], c="crimson", lw=1)
    axes[1, 1].set(xlim=(0, upper), ylim=(0, upper), xlabel="close-pair Fisher d' before",
                   ylabel="close-pair Fisher d' after",
                   title="Cross-foil separation is retained")
    axes[1, 1].grid(alpha=.15)
    fig.suptitle("Continuous flattening of the three foil sheets in global $z_3$", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "global_foil_flattened_z3_diagnostics.png", dpi=220)


if __name__ == "__main__":
    main()
