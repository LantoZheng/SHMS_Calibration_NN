"""Unsupervised mixture of local tangent Gaussian experts in raw 5D FP space.

A full-covariance Gaussian is the local tangent-patch approximation to a
Riemannian mixture.  No reconstructed sieve, foil, or existing-cluster fields
enter fitting; those fields are read afterward for diagnostics only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import RobustScaler

FEATURES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc"]
READ = FEATURES + ["foil_position", "cluster"]
SEED = 25521


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv"
    out = Path(__file__).parent / "results"; out.mkdir(exist_ok=True)
    df = pd.read_csv(path, usecols=READ).dropna().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.choice(len(df), 24000, replace=False)].reset_index(drop=True)
    x = RobustScaler(quantile_range=(5,95)).fit_transform(df[FEATURES])
    foil = df.foil_position.astype(int).to_numpy()
    reference = foil * 1000 + df.cluster.astype(int).to_numpy()
    records = []
    for components in (50, 100, 150, 220):
        model = GaussianMixture(n_components=components, covariance_type="full", reg_covar=2e-4,
                                max_iter=180, n_init=1, init_params="kmeans", random_state=SEED).fit(x)
        posterior = model.predict_proba(x)
        labels = posterior.argmax(axis=1)
        active = np.bincount(labels, minlength=components)
        # Effective local tangent dimension of each expert covariance.
        eig = np.linalg.eigvalsh(model.covariances_)
        effdim = (eig.sum(axis=1)**2) / np.maximum((eig**2).sum(axis=1), 1e-12)
        entropy = -np.sum(posterior*np.log(np.maximum(posterior, 1e-12)), axis=1)
        records.append({
            "components": components, "converged": bool(model.converged_), "iterations": int(model.n_iter_),
            "bic": float(model.bic(x)), "active_components": int((active > 0).sum()),
            "median_component_size": float(np.median(active[active > 0])), "max_component_fraction": float(active.max()/len(x)),
            "posterior_entropy_median": float(np.median(entropy)),
            "expert_effective_dimension_median": float(np.median(effdim)),
            "reference_hole_ami": float(adjusted_mutual_info_score(reference, labels)),
            "reference_hole_ari": float(adjusted_rand_score(reference, labels)),
            "reference_foil_ami": float(adjusted_mutual_info_score(foil, labels)),
        })
    scan = pd.DataFrame(records); scan.to_csv(out / "local_tangent_mixture_scan.csv", index=False)
    report = {
        "n_events": int(len(x)), "features": FEATURES,
        "method": "full-covariance Gaussian mixture: each component is a soft local tangent expert in raw robust-scaled 5D",
        "not_used_in_fit": ["sieve_x", "sieve_y", "foil_position", "cluster", "P_gtr_y", "cluster_center_x", "cluster_center_y"],
        "evaluation_only": "existing sieve-plane cluster and foil labels",
        "best_by_reference_ami": scan.sort_values(["reference_hole_ami", "reference_hole_ari"], ascending=False).head(1).to_dict(orient="records")[0],
        "best_by_bic": scan.sort_values("bic").head(1).to_dict(orient="records")[0],
        "verdict": "This tests whether local tangent patches naturally coincide with holes; no reconstructed coordinate is used to guide the experts."
    }
    (out / "local_tangent_mixture_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
