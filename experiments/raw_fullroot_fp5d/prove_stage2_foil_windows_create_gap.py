"""Show that the sharp foil gaps are imposed by the Stage-2 foil classifier.

The input skim spectrum is compared with the later labelled Stage-2 table.
Stage-2 foil labels are the only colours; the skim itself is unlabelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

SKIM = Path(r"C:\Users\Lanto\Desktop\AI_ML R-SIDIS\RootData\skimmed_shms_coin_replay_production_25521_-1.root")
STAGE2 = Path(r"C:\Users\Lanto\Desktop\AI_ML R-SIDIS\SHMS_Calibration_NN\dataset\stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv")
OUT = Path(__file__).parent / "results"
COLORS = ["#8EA9C9", "#EDBD7C", "#8DCCB7"]


def main():
    tree = uproot.open(SKIM)["T"]
    raw = tree.arrays(["P_gtr_y", "P_ngcer_npeSum", "P_hgcer_npeSum", "P_cal_etottracknorm"], library="np")
    base = np.isfinite(raw["P_gtr_y"]) & np.isfinite(raw["P_ngcer_npeSum"]) & np.isfinite(raw["P_hgcer_npeSum"]) & np.isfinite(raw["P_cal_etottracknorm"])
    base &= (raw["P_ngcer_npeSum"] >= 6) & (raw["P_hgcer_npeSum"] >= 0) & (raw["P_cal_etottracknorm"] >= .8) & (raw["P_cal_etottracknorm"] <= 1.8)
    skim_y = raw["P_gtr_y"][base]
    stage = pd.read_csv(STAGE2, usecols=["P_gtr_y", "foil_position"])
    bounds = stage.groupby("foil_position").P_gtr_y.agg(["count", "min", "max", "mean", "std"]).sort_index()
    intervals = [(float(bounds.loc[i, "min"]), float(bounds.loc[i, "max"])) for i in bounds.index]
    gaps = [{"between": f"foil {i} -> foil {i+1}", "low": intervals[i][1], "high": intervals[i+1][0], "width_cm": intervals[i+1][0] - intervals[i][1]} for i in range(len(intervals)-1)]
    bins = np.linspace(-3.2, 3.6, 69)

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.4), sharex=True, constrained_layout=True)
    axes[0].hist(skim_y, bins=bins, histtype="step", linewidth=1.8, color="0.25", density=True, label=f"skim + same PID (unlabelled): {len(skim_y):,}")
    axes[0].set(title="The skim spectrum itself remains continuous through the foil valleys", ylabel="normalised density")
    axes[0].legend(frameon=False); axes[0].grid(alpha=.16)
    for foil, color in zip(bounds.index, COLORS):
        y = stage.loc[stage.foil_position == foil, "P_gtr_y"]
        axes[1].hist(y, bins=bins, histtype="stepfilled", alpha=.62, density=True, color=color, label=f"Stage-2 foil {foil}: {len(y):,}")
        lo, hi = bounds.loc[foil, ["min", "max"]]
        axes[1].axvline(lo, color=color, linestyle="--", linewidth=1.2)
        axes[1].axvline(hi, color=color, linestyle="--", linewidth=1.2)
    for gap in gaps:
        axes[1].axvspan(gap["low"], gap["high"], color="0.25", alpha=.10)
        axes[1].annotate(f"hard gap\n{gap['width_cm']:.3f} cm", ((gap["low"]+gap["high"])/2, .12), ha="center", fontsize=9)
    axes[1].set(title="Sharp boundaries appear after `classify_foils_with_range(..., drop_unclassified=True)`", xlabel=r"reconstructed $y_{tar}$ ($P_{gtr,y}$)", ylabel="normalised density")
    axes[1].legend(frameon=False, ncol=3); axes[1].grid(alpha=.16)
    fig.suptitle("Run 25521: the observed sharp inter-foil cuts are Stage-2 classification windows, not skim track/delta cuts", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "stage2_foil_classification_windows_explain_ytar_gaps.png", dpi=230, bbox_inches="tight")
    report = {"skim_pid_events": int(len(skim_y)), "stage2_foil_bounds_cm": bounds.to_dict(orient="index"), "hard_gaps_cm": gaps,
              "provenance": "build_stage2_labels_from_25521_fullroot.py calls classify_foils_with_range(P_gtr_y, bins=50, sigma_factor=3, y_range=(-5,5), drop_unclassified=True)"}
    (OUT / "stage2_foil_classification_window_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
