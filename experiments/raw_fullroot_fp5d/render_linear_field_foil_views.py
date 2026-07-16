"""Render foil-like y_tar slices and vertical side views for the refined labels.

The third global coordinate is reconstructed P.gtr.y, rather than local z3:
local z3 values are only comparable inside their own sieve conflict cells.
Three event-level y_tar mixture components are visualization slices, not foil
truth labels and not inputs to either clustering stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = RESULTS / "linear_local_field_pipeline_labels.csv"
RANDOM_STATE = 25521


def colour(label: int) -> tuple[float, float, float, float]:
    return plt.get_cmap("hsv")((int(label) * 0.61803398875) % 1.0)


def main() -> None:
    raw = pd.read_csv(SOURCE)
    data = raw.loc[raw.linear_local_field_cluster >= 0].copy()
    ytar = data[["P.gtr.y"]].to_numpy()
    mixture = GaussianMixture(n_components=3, covariance_type="full", n_init=12, random_state=RANDOM_STATE).fit(ytar)
    order = np.argsort(mixture.means_.ravel())
    old_to_sorted = np.empty(3, dtype=int)
    old_to_sorted[order] = np.arange(3)
    posterior = mixture.predict_proba(ytar)
    slice_id = old_to_sorted[posterior.argmax(axis=1)]
    slice_confidence = posterior.max(axis=1)
    data["ytar_foil_slice"] = slice_id
    data["ytar_foil_slice_posterior"] = slice_confidence
    data.to_csv(RESULTS / "linear_local_field_ytar_foil_slices.csv", index=False)

    component_info = []
    for sorted_id, old_id in enumerate(order):
        selected = slice_id == sorted_id
        component_info.append({
            "slice": sorted_id, "ytar_mean_cm": float(mixture.means_[old_id, 0]),
            "ytar_sigma_cm": float(np.sqrt(mixture.covariances_[old_id, 0, 0])),
            "mixture_weight": float(mixture.weights_[old_id]), "argmax_events": int(selected.sum()),
            "median_posterior": float(np.median(slice_confidence[selected])),
        })
    (RESULTS / "linear_local_field_ytar_foil_slice_summary.json").write_text(json.dumps({
        "description": "Event-level 3-component GMM of reconstructed P.gtr.y for visualization only; not truth foil labels.",
        "coordinate": ["sieve_x", "sieve_y", "P.gtr.y"],
        "components": component_info,
    }, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex="col", constrained_layout=True)
    x_lim = (data.sieve_x.quantile(.002), data.sieve_x.quantile(.998))
    y_lim = (data.sieve_y.quantile(.002), data.sieve_y.quantile(.998))
    z_lim = (data["P.gtr.y"].quantile(.001), data["P.gtr.y"].quantile(.999))
    for foil in range(3):
        part = data.loc[data.ytar_foil_slice == foil]
        info = component_info[foil]
        top, side = axes[0, foil], axes[1, foil]
        # Keep rendering manageable while retaining each local cluster structure.
        draw = part.sample(min(35000, len(part)), random_state=RANDOM_STATE + foil)
        labels = draw.linear_local_field_cluster.to_numpy(dtype=int)
        top.scatter(draw.sieve_x, draw.sieve_y, c=[colour(label) for label in labels], s=.75, alpha=.62, linewidths=0, rasterized=True)
        top.set(xlim=x_lim, ylim=y_lim, title=(f"inferred foil slice {foil}: {len(part):,} clustered events\n"
                                                f"$y_{{tar}}$ GMM $\\mu$={info['ytar_mean_cm']:+.2f}$\\,$cm, $\\sigma$={info['ytar_sigma_cm']:.2f}$\\,$cm"),
                xlabel=r"reconstructed $x_{sieve}$")
        top.grid(alpha=.15)
        side.scatter(draw.sieve_x, draw["P.gtr.y"], c=[colour(label) for label in labels], s=.75, alpha=.62, linewidths=0, rasterized=True)
        side.axhline(info["ytar_mean_cm"], color="black", lw=.8, alpha=.6)
        side.set(xlim=x_lim, ylim=z_lim, title=r"vertical side view: $(x_{sieve},,y_{tar})$", xlabel=r"reconstructed $x_{sieve}$")
        side.grid(alpha=.15)
        if foil == 0:
            top.set_ylabel(r"reconstructed $y_{sieve}$")
            side.set_ylabel(r"reconstructed $y_{tar}$")
    fig.suptitle("Linear local-field clustering in global $(x_{sieve},y_{sieve},P.gtr.y)$ coordinates", fontsize=17, fontweight="bold")
    fig.savefig(RESULTS / "linear_local_field_foil_sieve_and_ytar_sideviews.png", dpi=220)


if __name__ == "__main__":
    main()
