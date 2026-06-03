#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(__file__).resolve().parent
OUT_DIR = TMP_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR = OUT_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ANALYZE_SCRIPT = REPO_ROOT / "training" / "scripts" / "analyze_stage2_ytar_distribution.py"

EXPERIMENTS = [
    {
        "name": "weaklabel_tolerant_base",
        "label_mode": "cluster-center weak labels",
        "changes": ["leave_one_foil_out", "cluster centers", "ytar tolerance>0", "no sep", "no noytar"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_general_gpu_clean" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis" / "ytar_distribution_summary.json",
    },
    {
        "name": "nearest_base",
        "label_mode": "nearest-hole-match",
        "changes": ["leave_one_foil_out", "hard foil centers", "no sep", "no noytar"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_nearest_hole_match_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis_mech25x16p4_tol3_nearest_hole_match_general_cuda" / "ytar_distribution_summary.json",
    },
    {
        "name": "nearest_sep_only",
        "label_mode": "nearest-hole-match",
        "changes": ["leave_one_foil_out", "hard foil centers", "sep=0.02", "no noytar"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3.csv",
        "source_summary": GENERATED_DIR / "nearest_sep_only" / "ytar_distribution_summary.json",
    },
    {
        "name": "nearest_sep_noytar",
        "label_mode": "nearest-hole-match",
        "changes": ["leave_one_foil_out", "hard foil centers", "sep=0.02", "noytar"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_noytar_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_noytar_general_cuda" / "ytar_distribution_summary.json",
    },
    {
        "name": "nearest_sep_noytar_delay2",
        "label_mode": "nearest-hole-match",
        "changes": ["leave_one_foil_out", "hard foil centers", "sep=0.02", "noytar", "delay2"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_noytar_delay2_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_noytar_delay2_general_cuda" / "ytar_distribution_summary.json",
    },
    {
        "name": "nearest_sep_noytar_delay2_allfoils",
        "label_mode": "nearest-hole-match",
        "changes": ["random split", "hard foil centers", "sep=0.02", "noytar", "delay2", "allfoils"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_noytar_delay2_allfoils_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis_mech25x16p4_tol3_nearest_hole_match_sep0.02t1_noytar_delay2_allfoils_general_cuda" / "ytar_distribution_summary.json",
    },
    {
        "name": "directgrid_max80_eqholeweight_fullnoytar",
        "label_mode": "directgrid+max80+eqholeweight",
        "changes": ["random split", "hard foil centers", "sep=0.02", "full noytar", "bs4096"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_directgrid_max80_eqholeweight_bs4096_fullnoytar_allfoils_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_directgrid_max80_eqholeweight.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis_mech25x16p4_tol3_directgrid_max80_eqholeweight_bs4096_fullnoytar_allfoils_general_cuda" / "ytar_distribution_summary.json",
    },
    {
        "name": "directgrid_max80_eqholeweight_sieveloss",
        "label_mode": "directgrid+max80+eqholeweight",
        "changes": ["random split", "hard foil centers", "sep=0.02", "full noytar", "sieve=0.35", "bs4096"],
        "checkpoint": REPO_ROOT / "checkpoints" / "stage2_transport_fullroot_25521_mech25x16p4_tol3_directgrid_max80_eqholeweight_bs4096_sieveloss035_fullnoytar_allfoils_general_cuda" / "best_finetune.pth",
        "data": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_directgrid_max80_eqholeweight.csv",
        "source_summary": REPO_ROOT / "outputs" / "stage2_ytar_analysis_mech25x16p4_tol3_directgrid_max80_eqholeweight_bs4096_sieveloss035_fullnoytar_allfoils_general_cuda" / "ytar_distribution_summary.json",
    },
]

COMPARISONS = [
    ("weak_to_hard_label_package", "weaklabel_tolerant_base", "nearest_base", "move from cluster-centered weak labels + tolerant ytar to mechanical-hole labels + hard ytar"),
    ("sep_only_vs_base", "nearest_base", "nearest_sep_only", "isolate sep on same labels/split"),
    ("noytar_vs_sep_only", "nearest_sep_only", "nearest_sep_noytar", "isolate noytar after sep on same labels/split"),
    ("delay2_vs_noytar", "nearest_sep_noytar", "nearest_sep_noytar_delay2", "isolate delay2 after noytar on same labels/split"),
    ("allfoils_random_vs_holdout", "nearest_sep_noytar_delay2", "nearest_sep_noytar_delay2_allfoils", "isolate removing foil holdout / using random all-foils split on same labels and training flow"),
    ("sieveloss_vs_fullnoytar", "directgrid_max80_eqholeweight_fullnoytar", "directgrid_max80_eqholeweight_sieveloss", "isolate sieve-plane loss on same labels/split"),
    ("compound_directgrid_vs_base", "nearest_base", "directgrid_max80_eqholeweight_fullnoytar", "compound effect of label/split/weighting/noytar"),
]


def ensure_summary(spec: dict[str, Any]) -> Path:
    summary_path = Path(spec["source_summary"])
    if summary_path.exists():
        return summary_path
    output_dir = GENERATED_DIR / spec["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--checkpoint",
        str(spec["checkpoint"]),
        "--data",
        str(spec["data"]),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    generated_summary = output_dir / "ytar_distribution_summary.json"
    if not generated_summary.exists():
        raise FileNotFoundError(f"Expected generated summary not found: {generated_summary}")
    return generated_summary



def load_summary(summary_path: Path) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_history_metrics(checkpoint_path: Path, checkpoint_epoch: int | None) -> dict[str, Any]:
    history_path = checkpoint_path.parent / "training_history_finetune.json"
    if not history_path.exists() or checkpoint_epoch is None:
        return {}

    with open(history_path, "r", encoding="utf-8") as fh:
        history = json.load(fh)

    idx = max(int(checkpoint_epoch) - 1, 0)

    def get_at(key: str) -> float | None:
        values = history.get(key)
        if not isinstance(values, list) or idx >= len(values):
            return None
        value = values[idx]
        return float(value) if value is not None else None

    x_dead = get_at("val_deadzone_rmse_xptar")
    y_dead = get_at("val_deadzone_rmse_yptar")
    return {
        "val_deadzone_rmse_xptar": x_dead,
        "val_deadzone_rmse_yptar": y_dead,
        "val_deadzone_rmse_ytar": get_at("val_deadzone_rmse_ytar"),
        "val_center_rmse_xptar": get_at("val_center_rmse_xptar"),
        "val_center_rmse_yptar": get_at("val_center_rmse_yptar"),
        "val_center_rmse_ytar": get_at("val_center_rmse_ytar"),
        "val_xy_deadzone_mean": float(0.5 * (x_dead + y_dead)) if x_dead is not None and y_dead is not None else None,
    }



def row_from_summary(spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    summary_map = {row["slice"]: row for row in payload["summaries"]}
    all_row = summary_map["all"]
    foil1_row = summary_map["foil1"]
    history_metrics = load_history_metrics(spec["checkpoint"], payload.get("checkpoint_epoch"))
    return {
        "experiment": spec["name"],
        "label_mode": spec["label_mode"],
        "changes": ", ".join(spec["changes"]),
        "summary_json": str(spec["source_summary"]),
        "checkpoint": str(spec["checkpoint"]),
        "data": str(spec["data"]),
        "checkpoint_val_loss": payload.get("checkpoint_val_loss"),
        "all_root_rmse_cm": all_row["root_rmse_to_center"],
        "all_nn_rmse_cm": all_row["nn_rmse_to_center"],
        "all_rmse_improvement_pct": all_row["rmse_improvement_pct"],
        "foil1_root_rmse_cm": foil1_row["root_rmse_to_center"],
        "foil1_nn_rmse_cm": foil1_row["nn_rmse_to_center"],
        "foil1_rmse_improvement_pct": foil1_row["rmse_improvement_pct"],
        "foil1_root_sigma68_cm": foil1_row["root_sigma68"],
        "foil1_nn_sigma68_cm": foil1_row["nn_sigma68"],
        "foil1_sigma68_improvement_pct": foil1_row["sigma68_improvement_pct"],
        "foil1_nn_bias_cm": foil1_row["nn_bias_to_center"],
        "foil1_root_bias_cm": foil1_row["root_bias_to_center"],
        **history_metrics,
    }



def impact_label(delta_cm: float) -> str:
    mag = abs(delta_cm)
    if mag < 0.05:
        return "small"
    if mag < 0.20:
        return "moderate"
    return "strong"



def compare_rows(name: str, base: pd.Series, cand: pd.Series, note: str) -> dict[str, Any]:
    foil1_delta = float(cand["foil1_nn_rmse_cm"] - base["foil1_nn_rmse_cm"])
    all_delta = float(cand["all_nn_rmse_cm"] - base["all_nn_rmse_cm"])
    sigma_delta = float(cand["foil1_nn_sigma68_cm"] - base["foil1_nn_sigma68_cm"])
    xy_dead_base = base.get("val_xy_deadzone_mean")
    xy_dead_cand = cand.get("val_xy_deadzone_mean")
    return {
        "comparison": name,
        "base": base["experiment"],
        "candidate": cand["experiment"],
        "note": note,
        "delta_foil1_nn_rmse_cm": foil1_delta,
        "delta_all_nn_rmse_cm": all_delta,
        "delta_foil1_nn_sigma68_cm": sigma_delta,
        "delta_val_xy_deadzone_mean": (float(xy_dead_cand) - float(xy_dead_base)) if pd.notna(xy_dead_base) and pd.notna(xy_dead_cand) else None,
        "impact_on_foil1_rmse": impact_label(foil1_delta),
        "direction": "worse" if foil1_delta > 0 else ("better" if foil1_delta < 0 else "same"),
    }



def main() -> None:
    rows: list[dict[str, Any]] = []
    payload_store: dict[str, dict[str, Any]] = {}

    for spec in EXPERIMENTS:
        summary_path = ensure_summary(spec)
        spec["source_summary"] = summary_path
        payload = load_summary(summary_path)
        payload_store[spec["name"]] = payload
        rows.append(row_from_summary(spec, payload))

    summary_df = pd.DataFrame(rows).sort_values("foil1_nn_rmse_cm").reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "stage2_effects_summary.csv", index=False)
    summary_df.to_json(OUT_DIR / "stage2_effects_summary.json", orient="records", force_ascii=False, indent=2)

    comparisons: list[dict[str, Any]] = []
    row_map = {row["experiment"]: row for _, row in summary_df.iterrows()}
    for name, base_name, cand_name, note in COMPARISONS:
        comparisons.append(compare_rows(name, row_map[base_name], row_map[cand_name], note))

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(OUT_DIR / "stage2_effect_comparisons.csv", index=False)
    comp_df.to_json(OUT_DIR / "stage2_effect_comparisons.json", orient="records", force_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    plot_df = summary_df.copy()
    plot_df.plot(kind="bar", x="experiment", y=["foil1_root_rmse_cm", "foil1_nn_rmse_cm"], ax=axes[0], color=["#7f7f7f", "#1f77b4"])
    axes[0].set_title("foil1 RMSE to hard center")
    axes[0].set_ylabel("RMSE [cm]")
    axes[0].grid(alpha=0.15, axis="y")
    axes[0].tick_params(axis="x", rotation=35)

    comp_df.plot(kind="bar", x="comparison", y="delta_foil1_nn_rmse_cm", ax=axes[1], color="#d62728")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Δ foil1 NN RMSE between paired experiments")
    axes[1].set_ylabel("Candidate - Base [cm]")
    axes[1].grid(alpha=0.15, axis="y")
    axes[1].tick_params(axis="x", rotation=35)

    fig.savefig(OUT_DIR / "stage2_effects_summary.png", dpi=180)
    plt.close(fig)

    print("Saved:")
    print(f"  {OUT_DIR / 'stage2_effects_summary.csv'}")
    print(f"  {OUT_DIR / 'stage2_effects_summary.json'}")
    print(f"  {OUT_DIR / 'stage2_effect_comparisons.csv'}")
    print(f"  {OUT_DIR / 'stage2_effect_comparisons.json'}")
    print(f"  {OUT_DIR / 'stage2_effects_summary.png'}")

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("\nExperiment summary:")
        print(summary_df[[
            "experiment",
            "label_mode",
            "val_deadzone_rmse_xptar",
            "val_deadzone_rmse_yptar",
            "val_xy_deadzone_mean",
            "foil1_root_rmse_cm",
            "foil1_nn_rmse_cm",
            "foil1_rmse_improvement_pct",
            "foil1_root_sigma68_cm",
            "foil1_nn_sigma68_cm",
            "foil1_sigma68_improvement_pct",
            "checkpoint_val_loss",
        ]].to_string(index=False))
        print("\nPaired comparisons:")
        print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
