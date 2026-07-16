"""Isolate which documented skim cuts sculpt the reconstructed ytar spectrum.

The comparison begins from full replay with the same SHMS PID cut and adds
only one documented skim condition at a time.  No sieve or foil labels enter.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

RAW = Path(r"C:\Users\Lanto\Desktop\AI_ML R-SIDIS\RootData\shms_coin_replay_production_25521_-1.root")
SKIM = Path(r"C:\Users\Lanto\Desktop\AI_ML R-SIDIS\RootData\skimmed_shms_coin_replay_production_25521_-1.root")
OUT = Path(__file__).parent / "results"
RAW_COLUMNS = ["P.gtr.y", "P.gtr.index", "P.gtr.dp", "P.ngcer.npeSum", "P.hgcer.npeSum", "P.cal.etottracknorm"]


def read_columns(path, columns):
    tree = uproot.open(path)["T"]
    available = set(tree.keys())
    use = [c for c in columns if c in available]
    return tree.arrays(use, library="np"), use


def hist(values, bins):
    return np.histogram(values, bins=bins)[0].astype(float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, use = read_columns(RAW, RAW_COLUMNS)
    finite = np.ones(len(raw["P.gtr.y"]), dtype=bool)
    for c in use: finite &= np.isfinite(raw[c])
    pid = (raw["P.ngcer.npeSum"] >= 6) & (raw["P.hgcer.npeSum"] >= 0) & (raw["P.cal.etottracknorm"] >= .8) & (raw["P.cal.etottracknorm"] <= 1.8)
    base = finite & pid
    cuts = {
        "full replay + PID": base,
        "+ P.gtr.index >= 0": base & (raw["P.gtr.index"] >= 0),
        "+ |P.gtr.dp| < 30": base & (np.abs(raw["P.gtr.dp"]) < 30),
        "+ index and |dp| < 30": base & (raw["P.gtr.index"] >= 0) & (np.abs(raw["P.gtr.dp"]) < 30),
    }
    # The skim stores the same quantities with underscore names.
    skim, skim_use = read_columns(SKIM, ["P_gtr_y", "P_gtr_index", "P_gtr_dp"])
    skim_finite = np.isfinite(skim["P_gtr_y"])
    bins = np.linspace(-3.25, 3.75, 71)
    centers = .5 * (bins[1:] + bins[:-1])
    colors = ["#808080", "#2E74B5", "#E07A35", "#A23B72"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), constrained_layout=True, sharex=True)
    for (name, mask), color in zip(cuts.items(), colors):
        axes[0, 0].hist(raw["P.gtr.y"][mask], bins=bins, histtype="step", linewidth=1.8, color=color, label=f"{name}: {mask.sum():,}")
    axes[0, 0].hist(skim["P_gtr_y"][skim_finite], bins=bins, histtype="step", linewidth=2.1, color="black", linestyle="--", label=f"actual skim: {skim_finite.sum():,}")
    axes[0, 0].set(title="One-cut-at-a-time ytar spectra", ylabel="events / bin")
    axes[0, 0].legend(fontsize=8, frameon=False); axes[0, 0].grid(alpha=.16)

    base_hist = hist(raw["P.gtr.y"][base], bins)
    for (name, mask), color in zip(list(cuts.items())[1:], colors[1:]):
        ratio = np.divide(hist(raw["P.gtr.y"][mask], bins), base_hist, out=np.zeros_like(base_hist), where=base_hist > 0)
        axes[0, 1].plot(centers, ratio, color=color, linewidth=1.7, label=name)
    axes[0, 1].axhline(1, color="0.45", linewidth=.8)
    axes[0, 1].set(title="Survival fraction relative to full replay + PID", ylabel="survival fraction", ylim=(-.05, 1.08))
    axes[0, 1].legend(fontsize=8, frameon=False); axes[0, 1].grid(alpha=.16)

    valid_index = base & (raw["P.gtr.index"] >= -1)
    index_values = sorted(np.unique(raw["P.gtr.index"][valid_index]).astype(int))
    for idx in index_values[:4]:
        m = valid_index & (raw["P.gtr.index"] == idx)
        axes[1, 0].hist(raw["P.gtr.y"][m], bins=bins, histtype="step", linewidth=1.5, label=f"P.gtr.index={idx}: {m.sum():,}")
    axes[1, 0].set(title="PID-selected ytar by track index", xlabel=r"reconstructed $y_{tar}$ ($P_{gtr,y}$)", ylabel="events / bin")
    axes[1, 0].legend(fontsize=8, frameon=False); axes[1, 0].grid(alpha=.16)

    dp_bins = np.linspace(-45, 45, 91)
    axes[1, 1].hist(raw["P.gtr.dp"][base], bins=dp_bins, histtype="step", linewidth=1.6, color="0.45", label="full replay + PID")
    axes[1, 1].hist(raw["P.gtr.dp"][base & (raw["P.gtr.index"] >= 0)], bins=dp_bins, histtype="step", linewidth=1.6, color="#2E74B5", label="+ index >= 0")
    axes[1, 1].axvline(-30, color="#E07A35", linestyle="--"); axes[1, 1].axvline(30, color="#E07A35", linestyle="--", label="SHMS singles |dp|<30")
    axes[1, 1].set(title="Does the loose SHMS delta cut intersect the PID sample?", xlabel=r"$P.gtr.dp$", ylabel="events / bin")
    axes[1, 1].legend(fontsize=8, frameon=False); axes[1, 1].grid(alpha=.16)
    fig.suptitle("Run 25521: diagnosing the skimmed-data ytar gap from documented cuts", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "skim_ytar_gap_one_cut_at_a_time.png", dpi=230, bbox_inches="tight")

    # Quantify which cut is ytar-dependent. A flat survival curve cannot open a foil gap.
    report = {"raw_entries": int(len(base)), "skim_entries_with_finite_ytar": int(skim_finite.sum()), "cuts": {}, "track_index_values_after_pid": {str(k): int((valid_index & (raw["P.gtr.index"] == k)).sum()) for k in index_values}}
    for name, mask in cuts.items():
        count = hist(raw["P.gtr.y"][mask], bins)
        survival = np.divide(count, base_hist, out=np.zeros_like(count), where=base_hist > 0)
        report["cuts"][name] = {"events": int(mask.sum()), "survival_min": float(survival[base_hist > 25].min()), "survival_max": float(survival[base_hist > 25].max()), "survival_std": float(survival[base_hist > 25].std())}
    (OUT / "skim_ytar_gap_cut_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
