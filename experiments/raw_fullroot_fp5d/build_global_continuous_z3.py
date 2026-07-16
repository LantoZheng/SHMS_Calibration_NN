"""Build one continuous z3 chart while allowing its FP5D normal to vary in z1,z2.

The existing flow z3 is retained as the global backbone.  A smooth ordinal
guide field g(u,f) is learned from the weak relative-foil labels, and only its
amplitude lambda(u) is allowed to vary over u=(flow_z1, flow_z2):

    z3_global(u,f) = flow_z3 + lambda(u) * g(u,f).

Both g and lambda use normalized Gaussian RBF partitions of unity, so this is a
single-valued continuous coordinate, not a collection of disconnected cells.
lambda is learned conservatively from close cross-foil cluster pairs to remove
the local nuisance direction that broadens flow_z3, while the identity term
prevents collapse of the already useful global flow coordinate.
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
from sklearn.preprocessing import RobustScaler, StandardScaler


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SOURCE = OUT / "nearest_foil_completed_labels.csv"
FP = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
U = ["flow_z1", "flow_z2"]
CLUSTER = "linear_local_field_cluster"
RANDOM_STATE = 25521
CLOSE_RADIUS = 0.30


def rbf(u: np.ndarray, centres: np.ndarray, width: float) -> np.ndarray:
    d2 = ((u[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    phi = np.exp(-0.5 * d2 / width**2)
    return phi / (phi.sum(axis=1, keepdims=True) + 1e-12)


def varying_linear_design(u: np.ndarray, f: np.ndarray, centres: np.ndarray, width: float) -> np.ndarray:
    phi = rbf(u, centres, width)
    augmented = np.column_stack([np.ones(len(f)), f])
    return (phi[:, :, None] * augmented[:, None, :]).reshape(len(f), -1)


def dprime(values: np.ndarray, binary: np.ndarray) -> float:
    a, b = values[binary == 0], values[binary == 1]
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)) + 1e-12)
    return float(abs(a.mean() - b.mean()) / pooled)


def main() -> None:
    raw = pd.read_csv(SOURCE)
    data = raw.loc[raw[CLUSTER] >= 0].copy()

    fp_scaler = RobustScaler(quantile_range=(5, 95)).fit(data[FP])
    f = fp_scaler.transform(data[FP])
    # Fit u scaling on cluster prototypes so every cluster has equal geometric weight.
    proto_u_raw = data.groupby(CLUSTER)[U].median()
    u_scaler = StandardScaler().fit(proto_u_raw)
    u = u_scaler.transform(data[U])

    work = data.assign(
        **{f"F{i}": f[:, i] for i in range(5)}, U0=u[:, 0], U1=u[:, 1]
    )
    prototype = work.groupby(CLUSTER).agg(
        **{f"F{i}": (f"F{i}", "median") for i in range(5)},
        U0=("U0", "median"), U1=("U1", "median"),
        foil=("final_relative_foil", "median"), events=(CLUSTER, "size"),
    )
    pu = prototype[["U0", "U1"]].to_numpy()
    pf = prototype[[f"F{i}" for i in range(5)]].to_numpy()

    # Smooth weak-prior guide.  It is deliberately not used as z3 by itself.
    guide_centres = KMeans(9, n_init=20, random_state=RANDOM_STATE).fit(pu).cluster_centers_
    guide_width = float(np.median(cKDTree(guide_centres).query(guide_centres, k=2)[0][:, 1]) * 1.45)
    guide_model = Ridge(alpha=0.03).fit(
        varying_linear_design(pu, pf, guide_centres, guide_width),
        prototype.foil.to_numpy(float) - 1.0,
    )
    guide = np.empty(len(data))
    for lo in range(0, len(data), 20_000):
        hi = min(lo + 20_000, len(data))
        guide[lo:hi] = guide_model.predict(
            varying_linear_design(u[lo:hi], f[lo:hi], guide_centres, guide_width)
        )
    work["global_ordinal_guide"] = guide

    # Close cross-foil cluster pairs define where the backbone needs correction.
    pairs: list[dict[str, float | int]] = []
    lambda_grid = np.linspace(-0.50, 0.20, 71)
    ids = prototype.index.to_numpy(int)
    for i in range(len(prototype)):
        for j in range(i + 1, len(prototype)):
            if prototype.foil.iloc[i] == prototype.foil.iloc[j]:
                continue
            distance = float(np.linalg.norm(pu[i] - pu[j]))
            if distance > CLOSE_RADIUS:
                continue
            ca, cb = int(ids[i]), int(ids[j])
            part = work.loc[work[CLUSTER].isin([ca, cb])]
            binary = (part[CLUSTER].to_numpy() == cb).astype(int)
            old = part.flow_z3.to_numpy()
            local_guide = part.global_ordinal_guide.to_numpy()
            scores = np.array([dprime(old + lam * local_guide, binary) for lam in lambda_grid])
            midpoint = 0.5 * (pu[i] + pu[j])
            pairs.append({
                "cluster_a": ca, "cluster_b": cb, "distance_z1z2": distance,
                "mid_u0": float(midpoint[0]), "mid_u1": float(midpoint[1]),
                "oracle_lambda": float(lambda_grid[np.argmax(scores)]),
                "dprime_flow_z3": dprime(old, binary), "dprime_oracle": float(scores.max()),
            })
    pair_df = pd.DataFrame(pairs)

    # Four broad gates and ridge shrinkage make lambda(u) smoother than the guide field.
    correction_centres = KMeans(4, n_init=20, random_state=RANDOM_STATE).fit(
        pair_df[["mid_u0", "mid_u1"]]
    ).cluster_centers_
    correction_width = float(
        np.median(cKDTree(correction_centres).query(correction_centres, k=2)[0][:, 1]) * 1.5
    )
    correction_model = Ridge(alpha=1.0).fit(
        rbf(pair_df[["mid_u0", "mid_u1"]].to_numpy(), correction_centres, correction_width),
        pair_df.oracle_lambda,
    )
    lambda_u = np.clip(rbf(u, correction_centres, correction_width) @ correction_model.coef_ + correction_model.intercept_, -0.50, 0.20)
    work["global_z3_correction_amplitude"] = lambda_u
    work["global_continuous_z3"] = work.flow_z3 + lambda_u * work.global_ordinal_guide

    # Pair metrics after applying the single global field.
    new_scores = []
    for row in pair_df.itertuples():
        part = work.loc[work[CLUSTER].isin([row.cluster_a, row.cluster_b])]
        binary = (part[CLUSTER].to_numpy() == row.cluster_b).astype(int)
        new_scores.append(dprime(part.global_continuous_z3.to_numpy(), binary))
    pair_df["dprime_global_continuous_z3"] = new_scores
    pair_df["dprime_gain"] = pair_df.dprime_global_continuous_z3 - pair_df.dprime_flow_z3
    pair_df.to_csv(OUT / "global_continuous_z3_close_pair_metrics.csv", index=False)

    # Preserve row alignment and avoid the duplicate-cluster merge issue of the exploratory script.
    output = raw.copy()
    for column in ["global_ordinal_guide", "global_z3_correction_amplitude", "global_continuous_z3"]:
        output[column] = np.nan
        output.loc[data.index, column] = work[column].to_numpy()
    output.to_csv(OUT / "global_continuous_z3_labels.csv", index=False)

    summary = {
        "definition": "z3_global = flow_z3 + lambda(flow_z1,flow_z2) * g(flow_z1,flow_z2,robust_scaled_FP5D)",
        "continuity": "g and lambda are normalized Gaussian-RBF partitions of unity; flow and all affine experts are continuous",
        "weak_supervision": "final relative foil ordering only; reconstructed ytar is not used",
        "close_pair_radius_standardized_z1z2": CLOSE_RADIUS,
        "close_cross_foil_pairs": int(len(pair_df)),
        "guide_experts": 9, "correction_gates": 4,
        "correction_lambda_quantiles": {
            "min": float(np.min(lambda_u)), "median": float(np.median(lambda_u)), "max": float(np.max(lambda_u))
        },
        "median_dprime_flow_z3": float(pair_df.dprime_flow_z3.median()),
        "median_dprime_global_continuous_z3": float(pair_df.dprime_global_continuous_z3.median()),
        "p10_dprime_flow_z3": float(pair_df.dprime_flow_z3.quantile(0.10)),
        "p10_dprime_global_continuous_z3": float(pair_df.dprime_global_continuous_z3.quantile(0.10)),
        "fraction_close_pairs_improved": float((pair_df.dprime_gain > 0).mean()),
        "mean_dprime_gain": float(pair_df.dprime_gain.mean()),
    }
    (OUT / "global_continuous_z3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Diagnostic plot: global chart, spatially varying correction, and separation gain.
    centres = work.groupby(CLUSTER).agg(
        z1=("flow_z1", "median"), z2=("flow_z2", "median"),
        z3=("global_continuous_z3", "median"), foil=("final_relative_foil", "median"),
        events=(CLUSTER, "size"),
    )
    colours = ["#356bb4", "#e98b27", "#35a16f"]
    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax_side = fig.add_subplot(2, 2, 2)
    ax_field = fig.add_subplot(2, 2, 3)
    ax_gain = fig.add_subplot(2, 2, 4)
    for foil in range(3):
        q = centres.loc[centres.foil == foil]
        size = np.clip(q.events / 5, 14, 100)
        ax3.scatter(q.z1, q.z2, q.z3, s=size, c=colours[foil], alpha=.82, label=f"foil {foil}")
        ax_side.scatter(q.z1, q.z3, s=size, c=colours[foil], alpha=.82, label=f"foil {foil}")
    ax3.set(xlabel="$z_1$", ylabel="$z_2$", zlabel="$z_3^{global}$", title="One global continuous chart")
    ax3.view_init(23, -58); ax3.legend(frameon=False)
    ax_side.set(xlabel="$z_1$", ylabel="$z_3^{global}$", title="Side view: foil layers remain globally ordered")
    ax_side.grid(alpha=.15)

    gx = np.linspace(pu[:, 0].min() - .15, pu[:, 0].max() + .15, 150)
    gy = np.linspace(pu[:, 1].min() - .15, pu[:, 1].max() + .15, 150)
    xx, yy = np.meshgrid(gx, gy); grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.clip(rbf(grid, correction_centres, correction_width) @ correction_model.coef_ + correction_model.intercept_, -.5, .2).reshape(xx.shape)
    contour = ax_field.contourf(xx, yy, zz, levels=18, cmap="coolwarm")
    ax_field.scatter(pu[:, 0], pu[:, 1], s=8, c="k", alpha=.32)
    fig.colorbar(contour, ax=ax_field, label="$\\lambda(z_1,z_2)$")
    ax_field.set(xlabel="standardized $z_1$", ylabel="standardized $z_2$", title="Continuous correction-amplitude field")

    upper = float(pair_df[["dprime_flow_z3", "dprime_global_continuous_z3"]].quantile(.98).max())
    ax_gain.scatter(pair_df.dprime_flow_z3, pair_df.dprime_global_continuous_z3,
                    c=pair_df.distance_z1z2, cmap="viridis", s=25, alpha=.8)
    ax_gain.plot([0, upper], [0, upper], c="crimson", lw=1)
    ax_gain.set(xlim=(0, upper), ylim=(0, upper), xlabel="Fisher d' in flow $z_3$",
                ylabel="Fisher d' in global continuous $z_3$",
                title=f"Close cross-foil pairs: {100*(pair_df.dprime_gain > 0).mean():.1f}% improved")
    ax_gain.grid(alpha=.15)
    fig.suptitle("Identity-anchored global $z_3$ with a smoothly varying local FP5D direction", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "global_continuous_z3_diagnostics.png", dpi=220)


if __name__ == "__main__":
    main()
