"""Construct normal coordinates for the three weakly supervised foil sheets.

The remapping learns a smooth scalar foliation C(z1,z2,z3) whose level sets
approximate the foil sheets.  Its output is the normal coordinate z3'.  The
remaining coordinates are obtained by transporting every event along the
gradient flow of C to the reference level C=0:

    d q / d C = grad C / ||grad C||^2,
    (z1', z2') = q_ref[C=0],  z3' = C(q).

Thus z3' lines follow the local normal field and the foil sheets become the
parallel planes z3'=-1,0,+1 in the ideal fit.  No event-level hard foil label
is needed once C has been trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SOURCE = OUT / "global_continuous_z3_labels.csv"
PAIR_SOURCE = OUT / "global_continuous_z3_close_pair_metrics.csv"
CLUSTER = "linear_local_field_cluster"
BASE_Z = "global_continuous_z3"
FP = ["flow_z1", "flow_z2"]
RANDOM_STATE = 25521
N_EXPERTS = 36
RIDGE_ALPHA = 0.01
FLOW_STEPS = 20


def rbf_and_derivative(u: np.ndarray, centres: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized RBF gates and d(gate)/d(u), shape n,k and n,k,2."""
    delta = u[:, None, :] - centres[None, :, :]
    unnormalized = np.exp(-0.5 * (delta**2).sum(axis=2) / width**2)
    phi = unnormalized / (unnormalized.sum(axis=1, keepdims=True) + 1e-12)
    raw_grad = -delta / width**2
    mean_grad = (phi[:, :, None] * raw_grad).sum(axis=1, keepdims=True)
    return phi, phi[:, :, None] * (raw_grad - mean_grad)


def dprime(values: np.ndarray, binary: np.ndarray) -> float:
    first, second = values[binary == 0], values[binary == 1]
    return float(abs(first.mean() - second.mean()) / np.sqrt(0.5 * (first.var(ddof=1) + second.var(ddof=1)) + 1e-12))


def plane_fit_metrics(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """RMSE of z=a+b*x+c*y, gradient norm, and unit plane normal."""
    design = np.column_stack([np.ones(len(x)), x[:, :2]])
    coefficients = np.linalg.lstsq(design, x[:, 2], rcond=None)[0]
    residual = x[:, 2] - design @ coefficients
    normal = np.array([-coefficients[1], -coefficients[2], 1.0])
    normal /= np.linalg.norm(normal)
    return float(np.sqrt(np.mean(residual**2))), float(np.linalg.norm(coefficients[1:])), normal


def main() -> None:
    raw = pd.read_csv(SOURCE)
    data = raw.loc[raw[CLUSTER] >= 0].copy()
    prototype = data.groupby(CLUSTER).agg(
        z1=("flow_z1", "median"), z2=("flow_z2", "median"),
        z3=(BASE_Z, "median"), foil=("final_relative_foil", "median"),
        events=(CLUSTER, "size"),
    )

    u_scaler = StandardScaler().fit(prototype[["z1", "z2"]].to_numpy())
    u_proto = u_scaler.transform(prototype[["z1", "z2"]].to_numpy())
    u_event = u_scaler.transform(data[FP].to_numpy())
    z_scale = float(np.std(prototype.z3.to_numpy()))
    q3_proto = prototype.z3.to_numpy() / z_scale
    q3_event = data[BASE_Z].to_numpy() / z_scale

    centres = KMeans(N_EXPERTS, n_init=20, random_state=RANDOM_STATE).fit(u_proto).cluster_centers_
    width = float(np.median(cKDTree(centres).query(centres, k=2)[0][:, 1]) * 1.5)
    phi_proto, _ = rbf_and_derivative(u_proto, centres, width)
    x_proto = (phi_proto[:, :, None] * np.column_stack([np.ones(len(q3_proto)), q3_proto])[:, None, :]).reshape(len(prototype), -1)
    target = prototype.foil.to_numpy(float) - 1.0

    cv_prediction = np.empty(len(prototype))
    for train, test in KFold(5, shuffle=True, random_state=RANDOM_STATE).split(x_proto):
        cv_prediction[test] = Ridge(alpha=RIDGE_ALPHA).fit(x_proto[train], target[train]).predict(x_proto[test])
    model = Ridge(alpha=RIDGE_ALPHA).fit(x_proto, target)
    coefficients = model.coef_.reshape(N_EXPERTS, 2)

    def scalar_and_gradient(u: np.ndarray, q3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi, dphi = rbf_and_derivative(u, centres, width)
        local_value = coefficients[:, 0][None, :] + q3[:, None] * coefficients[:, 1][None, :]
        value = model.intercept_ + (phi * local_value).sum(axis=1)
        gradient_u = (dphi * local_value[:, :, None]).sum(axis=1)
        gradient_q3 = phi @ coefficients[:, 1]
        return value, np.column_stack([gradient_u, gradient_q3])

    c_event, _ = scalar_and_gradient(u_event, q3_event)
    # Gradient-flow transport to the C=0 middle-foil surface.
    q_transport = np.column_stack([u_event, q3_event]).copy()
    scheduled_levels = c_event[:, None] * np.linspace(1.0 - 1.0 / FLOW_STEPS, 0.0, FLOW_STEPS)[None, :]
    for step in range(FLOW_STEPS):
        current_c, gradient = scalar_and_gradient(q_transport[:, :2], q_transport[:, 2])
        delta_c = scheduled_levels[:, step] - current_c
        q_transport += delta_c[:, None] * gradient / ((gradient**2).sum(axis=1, keepdims=True) + 1e-12)
    transport_residual, _ = scalar_and_gradient(q_transport[:, :2], q_transport[:, 2])

    reference_u = q_transport[:, :2]
    reference_original = u_scaler.inverse_transform(reference_u)
    data["foliation_z1"] = reference_original[:, 0]
    data["foliation_z2"] = reference_original[:, 1]
    data["foliation_z3"] = c_event
    data["foliation_reference_residual"] = transport_residual

    output = raw.copy()
    for column in ["foliation_z1", "foliation_z2", "foliation_z3", "foliation_reference_residual"]:
        output[column] = np.nan
        output.loc[data.index, column] = data[column].to_numpy()
    output.to_csv(OUT / "normal_foliation_remap_labels.csv", index=False)

    cluster_metrics = data.groupby(CLUSTER).agg(
        z1=("flow_z1", "median"), z2=("flow_z2", "median"), z3=(BASE_Z, "median"),
        z1_prime=("foliation_z1", "median"), z2_prime=("foliation_z2", "median"), z3_prime=("foliation_z3", "median"),
        foil=("final_relative_foil", "median"), events=(CLUSTER, "size"),
    )
    cluster_metrics.to_csv(OUT / "normal_foliation_remap_cluster_metrics.csv")

    pair_metrics = pd.read_csv(PAIR_SOURCE)
    foliation_dprime = []
    for row in pair_metrics.itertuples():
        part = data.loc[data[CLUSTER].isin([row.cluster_a, row.cluster_b])]
        binary = (part[CLUSTER].to_numpy() == row.cluster_b).astype(int)
        foliation_dprime.append(dprime(part.foliation_z3.to_numpy(), binary))
    pair_metrics["dprime_normal_foliation_z3"] = foliation_dprime
    pair_metrics["normal_foliation_dprime_gain"] = pair_metrics.dprime_normal_foliation_z3 - pair_metrics.dprime_global_continuous_z3
    pair_metrics.to_csv(OUT / "normal_foliation_remap_close_pair_metrics.csv", index=False)

    before_metrics, after_metrics = {}, {}
    before_normals, after_normals = [], []
    for foil in range(3):
        before = cluster_metrics.loc[cluster_metrics.foil == foil, ["z1", "z2", "z3"]].to_numpy()
        after = cluster_metrics.loc[cluster_metrics.foil == foil, ["z1_prime", "z2_prime", "z3_prime"]].to_numpy()
        before_rmse, before_slope, before_normal = plane_fit_metrics(before)
        after_rmse, after_slope, after_normal = plane_fit_metrics(after)
        before_metrics[str(foil)] = {"plane_rmse": before_rmse, "surface_slope": before_slope}
        after_metrics[str(foil)] = {"plane_rmse": after_rmse, "surface_slope": after_slope}
        before_normals.append(before_normal); after_normals.append(after_normal)
    normal_angles_before, normal_angles_after = [], []
    for i in range(3):
        for j in range(i + 1, 3):
            normal_angles_before.append(float(np.degrees(np.arccos(np.clip(abs(before_normals[i] @ before_normals[j]), -1, 1)))))
            normal_angles_after.append(float(np.degrees(np.arccos(np.clip(abs(after_normals[i] @ after_normals[j]), -1, 1)))))

    summary = {
        "definition": "z3_prime=C(z1,z2,z3); (z1_prime,z2_prime) are the C=0 footpoint obtained by normalized gradient flow",
        "training_weak_supervision": "cluster-level final relative foil rank only; reconstructed ytar excluded",
        "inference_requires_foil_label": False,
        "continuity": "C is a normalized Gaussian-RBF varying-affine field; gradient transport is continuous",
        "normal_coordinate_property": "transport paths follow grad(C), which is orthogonal to every C=constant sheet",
        "experts": N_EXPERTS, "ridge_alpha": RIDGE_ALPHA, "rbf_width": width, "z3_input_scale": z_scale,
        "cluster_5fold_mae": float(np.mean(abs(cv_prediction - target))),
        "cluster_5fold_nearest_foil_accuracy": float(np.mean(np.rint(np.clip(cv_prediction, -1, 1)) == target)),
        "minimum_vertical_foliation_slope": float((rbf_and_derivative(u_event, centres, width)[0] @ coefficients[:, 1]).min()),
        "max_abs_reference_surface_residual": float(np.max(abs(transport_residual))),
        "plane_fit_before": before_metrics,
        "plane_fit_after": after_metrics,
        "mean_pairwise_sheet_normal_angle_deg_before": float(np.mean(normal_angles_before)),
        "mean_pairwise_sheet_normal_angle_deg_after": float(np.mean(normal_angles_after)),
        "close_cross_foil_pairs": {
            "pairs": int(len(pair_metrics)),
            "median_dprime_before": float(pair_metrics.dprime_global_continuous_z3.median()),
            "median_dprime_after": float(pair_metrics.dprime_normal_foliation_z3.median()),
            "p10_dprime_before": float(pair_metrics.dprime_global_continuous_z3.quantile(.10)),
            "p10_dprime_after": float(pair_metrics.dprime_normal_foliation_z3.quantile(.10)),
        },
    }
    (OUT / "normal_foliation_remap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colours = ["#356bb4", "#e98b27", "#35a16f"]
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    ax_before = fig.add_subplot(2, 2, 1, projection="3d")
    ax_after = fig.add_subplot(2, 2, 2, projection="3d")
    ax_slope = fig.add_subplot(2, 2, 3)
    ax_pair = fig.add_subplot(2, 2, 4)
    for foil in range(3):
        q = cluster_metrics.loc[cluster_metrics.foil == foil]
        sizes = np.clip(q.events / 5, 14, 100)
        ax_before.scatter(q.z1, q.z2, q.z3, s=sizes, c=colours[foil], alpha=.8, label=f"foil {foil}")
        ax_after.scatter(q.z1_prime, q.z2_prime, q.z3_prime, s=sizes, c=colours[foil], alpha=.8, label=f"foil {foil}")
    ax_before.set(xlabel="$z_1$", ylabel="$z_2$", zlabel="$z_3$", title="Current global coordinates")
    ax_after.set(xlabel="$z'_1$", ylabel="$z'_2$", zlabel="$z'_3$", title="Gradient-flow normal coordinates")
    for ax in [ax_before, ax_after]: ax.view_init(23, -58); ax.legend(frameon=False)
    ax_slope.bar(np.arange(3) - .18, [before_metrics[str(f)]["surface_slope"] for f in range(3)], .35, color="#8ba7cf", label="before")
    ax_slope.bar(np.arange(3) + .18, [after_metrics[str(f)]["surface_slope"] for f in range(3)], .35, color="#46a979", label="after")
    ax_slope.set(xticks=np.arange(3), xticklabels=["foil 0", "foil 1", "foil 2"], ylabel="best-plane slope magnitude", title="Sheet tilt in its own coordinates")
    ax_slope.legend(frameon=False); ax_slope.grid(axis="y", alpha=.15)
    lim = float(pair_metrics[["dprime_global_continuous_z3", "dprime_normal_foliation_z3"]].quantile(.98).max())
    ax_pair.scatter(pair_metrics.dprime_global_continuous_z3, pair_metrics.dprime_normal_foliation_z3, c=pair_metrics.distance_z1z2, cmap="viridis", s=24, alpha=.8)
    ax_pair.plot([0, lim], [0, lim], c="crimson", lw=1)
    ax_pair.set(xlim=(0, lim), ylim=(0, lim), xlabel="close-pair Fisher d' before", ylabel="close-pair Fisher d' after", title="Cross-foil separation under the new normal coordinate")
    ax_pair.grid(alpha=.15)
    fig.suptitle("Normal-coordinate remapping: foil sheets as parallel z′3 planes", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "normal_foliation_remap_diagnostics.png", dpi=220)


if __name__ == "__main__":
    main()
