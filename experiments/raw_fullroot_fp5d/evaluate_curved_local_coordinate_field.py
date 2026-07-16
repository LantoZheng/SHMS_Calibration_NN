"""Give a local branch coordinate freedom to bend through (z3,z4,z5).

Within each conflict cell, robust-scaled FP5D is represented by a local PCA
triplet u=(z3,z4,z5).  A quadratic logistic score s(u) is the resolved
coordinate.  Its s=0 boundary is a curved surface, so the apparent z3 tangent
can rotate as z4,z5 change.  HDBSCAN high-probability members are weak seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FEATURES = ["P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.rb.raster.frybRawAdc"]
PAIRS = [("S1", 219, 220), ("S2", 15, 68), ("S3", 2, 3), ("S4", 45, 56)]
SEED_PROBABILITY = 0.80


def symmetric_auc(score: np.ndarray, target: np.ndarray) -> float:
    auc = roc_auc_score(target, score)
    return float(max(auc, 1 - auc))


def main() -> None:
    df = pd.read_csv(RESULTS / "raw_fullroot_flow_hdbscan_labels.csv")
    data = df.loc[df.flow_hdbscan_cluster >= 0].copy()
    fp = RobustScaler(quantile_range=(5, 95)).fit_transform(data[FEATURES])
    fp = pd.DataFrame(fp, index=data.index, columns=FEATURES)
    linear_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    curved_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", random_state=25521),
    )
    metrics, output = [], []
    fig = plt.figure(figsize=(21, 10), constrained_layout=True)
    axes3d, axes2d = [], []
    for index in range(4):
        axes3d.append(fig.add_subplot(2, 4, index + 1, projection="3d"))
        axes2d.append(fig.add_subplot(2, 4, 4 + index + 1))

    for col, (tag, first, second) in enumerate(PAIRS):
        part = data.loc[data.flow_hdbscan_cluster.isin([first, second])].copy()
        target = (part.flow_hdbscan_cluster.to_numpy() == second).astype(int)
        x = fp.loc[part.index].to_numpy()
        seeds = part.hdbscan_probability.to_numpy() >= SEED_PROBABILITY
        if min(np.bincount(target[seeds], minlength=2)) < 20:
            seeds[:] = True
        # Local coordinates are intentionally re-estimated in each sieve cell.
        pca = PCA(n_components=3, whiten=True, random_state=25521).fit(x[seeds])
        u = pca.transform(x)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=25521)
        linear_oof = cross_val_predict(linear_model, x[seeds], target[seeds], cv=cv, method="decision_function")
        curved_oof = cross_val_predict(curved_model, u[seeds], target[seeds], cv=cv, method="decision_function")
        fitted = curved_model.fit(u[seeds], target[seeds])
        curved_score = fitted.decision_function(u)
        if part.loc[target == 1, "P.gtr.y"].mean() < part.loc[target == 0, "P.gtr.y"].mean():
            curved_score *= -1
        part["curved_field_z3"] = curved_score
        part["curved_field_z4"] = u[:, 1]
        part["curved_field_z5"] = u[:, 2]
        part["curved_field_seed"] = seeds
        output.append(part)
        linear_auc, curved_auc = symmetric_auc(linear_oof, target[seeds]), symmetric_auc(curved_oof, target[seeds])
        metrics.append({
            "pair": tag, "clusters": f"{first} vs {second}", "seed_events": int(seeds.sum()),
            "linear_fp5d_oof_auc": linear_auc, "curved_z3z4z5_oof_auc": curved_auc,
            "curved_minus_linear_auc": curved_auc - linear_auc,
            "local_pca_variance": pca.explained_variance_ratio_.tolist(),
        })

        ax3 = axes3d[col]
        for branch, colour, label in [(0, "tab:blue", str(first)), (1, "tab:orange", str(second))]:
            mask = target == branch
            ax3.scatter(u[mask, 0], u[mask, 1], u[mask, 2], s=11, alpha=.75, color=colour, label=label)
        ax3.set(title=f"{tag}: local $(z_3,z_4,z_5)$\nlinear AUC={linear_auc:.3f}, curved={curved_auc:.3f}", xlabel=r"$z_3$", ylabel=r"$z_4$", zlabel=r"$z_5$")
        ax3.view_init(elev=23, azim=-58)
        ax3.legend(frameon=False, fontsize=8)

        ax2 = axes2d[col]
        for branch, colour, label in [(0, "tab:blue", str(first)), (1, "tab:orange", str(second))]:
            mask = target == branch
            ax2.scatter(u[mask, 1], curved_score[mask], s=12, alpha=.73, color=colour, label=label, linewidths=0)
        ax2.axhline(0, color="black", lw=.8, alpha=.55)
        ax2.set(title=r"curved $z_3=g(z_4,z_5)$ branch coordinate", xlabel=r"local $z_4$", ylabel=r"curved local $z_3$")
        ax2.grid(alpha=.16)
        if col == 0: ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Curved local coordinate fields: let the branch-normal direction vary with $z_4$ and $z_5$", fontsize=17, fontweight="bold")
    fig.savefig(RESULTS / "curved_local_coordinate_field_z3z4z5.png", dpi=200)
    pd.DataFrame(metrics).to_csv(RESULTS / "curved_local_coordinate_field_metrics.csv", index=False)
    pd.concat(output, ignore_index=True).to_csv(RESULTS / "curved_local_coordinate_field_events.csv", index=False)
    (RESULTS / "curved_local_coordinate_field_method.json").write_text(json.dumps({
        "local_basis": "u=(z3,z4,z5)=whitened PCA(FP5D), fitted independently in each conflict cell from high-confidence seeds",
        "curved_coordinate": "s(u)=beta0 + sum beta_i u_i + sum beta_ij u_i*u_j; s=0 is the local curved branch surface",
        "interpretation": "the local normal/branch coordinate can rotate and bend with z4,z5; values remain local to each sieve neighbourhood",
        "evaluation": "five-fold OOF AUC over seed events; it remains weakly supervised by existing density-cluster seeds",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
