#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOIL_CENTERS = {0: 10.0, 1: 0.0, 2: -10.0}
DATASETS = [
    {
        "name": "original_weak_labels",
        "path": REPO_ROOT / "dataset" / "stage2_25521_labeled.csv",
        "label_kind": "weak-centers",
    },
    {
        "name": "mech25x16p4_tol3",
        "path": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3.csv",
        "label_kind": "hard-centers",
    },
    {
        "name": "directgrid",
        "path": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_directgrid.csv",
        "label_kind": "hard-centers",
    },
    {
        "name": "directgrid_max80_eqholeweight",
        "path": REPO_ROOT / "dataset" / "stage2_25521_labeled_mech25x16p4_tol3_directgrid_max80_eqholeweight.csv",
        "label_kind": "hard-centers",
    },
]

TARGET_COLS = [
    "foil_position",
    "P_react_z",
    "weak_foil_ytar_center",
    "weak_foil_ytar_tol",
    "foil_ytar_center",
    "foil_ytar_tol",
]


def nearest_center_accuracy(values: np.ndarray, true_foil: np.ndarray) -> float:
    centers = np.array([FOIL_CENTERS[idx] for idx in sorted(FOIL_CENTERS)], dtype=np.float64)
    pred = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    return float(np.mean(pred == true_foil.astype(int)))



def summarize_alignment(df: pd.DataFrame, values: np.ndarray, expected: np.ndarray, sign_name: str) -> dict[str, Any]:
    residual = values - expected
    result: dict[str, Any] = {
        "sign_mode": sign_name,
        "all_rmse_cm": float(np.sqrt(np.mean(residual**2))),
        "all_mae_cm": float(np.mean(np.abs(residual))),
        "all_bias_cm": float(np.mean(residual)),
        "nearest_center_accuracy": nearest_center_accuracy(values, df["foil_position"].to_numpy(dtype=np.int64)),
        "per_foil": [],
    }
    for foil in sorted(df["foil_position"].dropna().unique()):
        foil_int = int(foil)
        dff = df.loc[df["foil_position"] == foil_int]
        expected_foil = np.full(len(dff), FOIL_CENTERS[foil_int], dtype=np.float64)
        value_foil = values[dff.index.to_numpy()]
        residual_foil = value_foil - expected_foil
        result["per_foil"].append(
            {
                "foil": foil_int,
                "n_events": int(len(dff)),
                "expected_center_cm": float(FOIL_CENTERS[foil_int]),
                "mean_value_cm": float(np.mean(value_foil)),
                "median_value_cm": float(np.median(value_foil)),
                "std_value_cm": float(np.std(value_foil)),
                "rmse_to_center_cm": float(np.sqrt(np.mean(residual_foil**2))),
                "mae_to_center_cm": float(np.mean(np.abs(residual_foil))),
                "bias_to_center_cm": float(np.mean(residual_foil)),
            }
        )
    return result



def extract_label_center_report(df: pd.DataFrame) -> dict[str, Any]:
    label_cols = [col for col in ["foil_ytar_center", "weak_foil_ytar_center"] if col in df.columns]
    tol_cols = [col for col in ["foil_ytar_tol", "weak_foil_ytar_tol"] if col in df.columns]
    report: dict[str, Any] = {"label_columns_present": label_cols, "tol_columns_present": tol_cols, "per_foil": []}
    for foil in sorted(df["foil_position"].dropna().unique()):
        foil_int = int(foil)
        dff = df.loc[df["foil_position"] == foil_int]
        foil_row: dict[str, Any] = {
            "foil": foil_int,
            "expected_center_cm": float(FOIL_CENTERS[foil_int]),
        }
        for col in label_cols:
            uniques = sorted(float(v) for v in pd.unique(dff[col].dropna()))
            foil_row[f"{col}_unique"] = uniques
            foil_row[f"{col}_matches_expected"] = bool(len(uniques) == 1 and np.isclose(uniques[0], FOIL_CENTERS[foil_int], atol=1e-9))
        for col in tol_cols:
            uniques = sorted(float(v) for v in pd.unique(dff[col].dropna()))
            foil_row[f"{col}_unique"] = uniques
        report["per_foil"].append(foil_row)
    return report



def load_dataset(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in TARGET_COLS if col in header.columns]
    return pd.read_csv(path, usecols=usecols)



def main() -> None:
    dataset_reports: list[dict[str, Any]] = []
    rmse_plot_rows: list[dict[str, Any]] = []
    per_foil_rows: list[dict[str, Any]] = []

    for spec in DATASETS:
        path = spec["path"]
        if not path.exists():
            continue
        df = load_dataset(path)
        df = df.dropna(subset=["foil_position", "P_react_z"]).copy()
        df["foil_position"] = df["foil_position"].astype(int)
        expected = df["foil_position"].map(FOIL_CENTERS).to_numpy(dtype=np.float64)
        raw = df["P_react_z"].to_numpy(dtype=np.float64)
        flipped = -raw

        label_report = extract_label_center_report(df)
        raw_report = summarize_alignment(df, raw, expected, "P_react_z")
        flipped_report = summarize_alignment(df, flipped, expected, "-P_react_z")

        dataset_report = {
            "dataset": spec["name"],
            "path": str(path),
            "label_kind": spec["label_kind"],
            "n_events": int(len(df)),
            "hard_foil_centers_cm": FOIL_CENTERS,
            "label_center_report": label_report,
            "alignment": [raw_report, flipped_report],
        }
        dataset_reports.append(dataset_report)

        for align in dataset_report["alignment"]:
            rmse_plot_rows.append(
                {
                    "dataset": spec["name"],
                    "sign_mode": align["sign_mode"],
                    "all_rmse_cm": align["all_rmse_cm"],
                    "all_mae_cm": align["all_mae_cm"],
                    "all_bias_cm": align["all_bias_cm"],
                    "nearest_center_accuracy": align["nearest_center_accuracy"],
                }
            )
            for foil_row in align["per_foil"]:
                per_foil_rows.append({"dataset": spec["name"], "sign_mode": align["sign_mode"], **foil_row})

    if not dataset_reports:
        raise SystemExit("No datasets found to analyze.")

    rmse_df = pd.DataFrame(rmse_plot_rows)
    per_foil_df = pd.DataFrame(per_foil_rows)
    rmse_df.to_csv(OUT_DIR / "ytar_foil_z_check.csv", index=False)
    per_foil_df.to_csv(OUT_DIR / "ytar_foil_z_check_per_foil.csv", index=False)

    with open(OUT_DIR / "ytar_foil_z_check.json", "w", encoding="utf-8") as fh:
        json.dump(dataset_reports, fh, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    pivot_rmse = rmse_df.pivot(index="dataset", columns="sign_mode", values="all_rmse_cm")
    pivot_acc = rmse_df.pivot(index="dataset", columns="sign_mode", values="nearest_center_accuracy")

    pivot_rmse.plot(kind="bar", ax=axes[0], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("RMSE to hard foil centers")
    axes[0].set_ylabel("RMSE [cm]")
    axes[0].grid(alpha=0.15, axis="y")
    axes[0].legend(title="sign mode")

    pivot_acc.plot(kind="bar", ax=axes[1], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Nearest hard-center foil accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.15, axis="y")
    axes[1].legend(title="sign mode")

    fig.savefig(OUT_DIR / "ytar_foil_z_check.png", dpi=180)
    plt.close(fig)

    print("Saved:")
    print(f"  {OUT_DIR / 'ytar_foil_z_check.json'}")
    print(f"  {OUT_DIR / 'ytar_foil_z_check.csv'}")
    print(f"  {OUT_DIR / 'ytar_foil_z_check_per_foil.csv'}")
    print(f"  {OUT_DIR / 'ytar_foil_z_check.png'}")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nOverall alignment summary:")
        print(rmse_df.to_string(index=False))


if __name__ == "__main__":
    main()
