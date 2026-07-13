"""Evaluate NN and XGBoost checkpoints on the exact same held-out holes."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
XGB_ROOT = WORKSPACE / "Calibration_XGBoost"
sys.path.insert(0, str(ROOT / "src"))

from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.models import build_model_from_config


TARGETS = ("delta", "xptar", "yptar", "ytar")


def metrics(pred: np.ndarray, target: np.ndarray, tol: np.ndarray, mask: np.ndarray, scales: np.ndarray) -> dict:
    result: dict[str, float] = {}
    for i, name in enumerate(TARGETS):
        valid = mask[:, i] > 0.5
        error = (pred[valid, i] - target[valid, i]) * scales[i]
        tolerance = tol[valid, i] * scales[i]
        result[f"{name}_center_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        result[f"{name}_center_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_deadzone_rmse"] = float(np.sqrt(np.mean(np.maximum(np.abs(error) - tolerance, 0.0) ** 2)))
        result[f"{name}_within_tol"] = float(np.mean(np.abs(error) <= tolerance))
    return result


def evaluate_by_foil(pred: np.ndarray, target: np.ndarray, tol: np.ndarray, mask: np.ndarray, scales: np.ndarray, foils: np.ndarray) -> dict:
    return {str(int(foil)): metrics(pred[foils == foil], target[foils == foil], tol[foils == foil], mask[foils == foil], scales)
            for foil in sorted(np.unique(foils))}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "common_holes_v3.yaml").read_text(encoding="utf-8"))
    scaler = ScalerBundle.load(ROOT / "models" / "scaler_bundle.json")
    dcfg = cfg["data"]
    dataset = Stage2RootDataset(
        ROOT / "data" / "stage2_soc_gui_labeled.csv", tree_name=dcfg["tree_name"], scaler_bundle=scaler,
        feature_schema=dcfg["feature_schema"], branch_map=dcfg["branch_map"], label_map=dcfg["label_map"],
        metadata_cols=dcfg["metadata_cols"], weight_col=dcfg["weight_col"], fry_mode=dcfg["fry_mode"],
        direct_fry_branch=dcfg["direct_fry_branch"], fry_proxy_branches=dcfg["fry_proxy_branches"], cuts=dcfg["cuts"],
    )
    holdout = [int(v) for v in cfg["validation"]["holdout_holes"]]
    indices = np.flatnonzero(dataset.metadata["hole_id"].isin(holdout).to_numpy())
    x = dataset.X[indices].numpy()
    target = np.column_stack([dataset.targets[key][indices].numpy().ravel() for key in TARGETS])
    tol = np.column_stack([dataset.tolerances[key][indices].numpy().ravel() for key in TARGETS])
    mask = np.column_stack([dataset.target_mask[key][indices].numpy().ravel() for key in TARGETS])
    foils = dataset.metadata.iloc[indices]["foil_position"].to_numpy(dtype=np.int32)
    scales = scaler.scaler_Y.scale_.astype(np.float64)

    nn_ckpt = torch.load(ROOT / "models" / "common_holes_v3" / "best_finetune.pth", map_location="cuda", weights_only=False)
    model = build_model_from_config(nn_ckpt["config"]["model"], input_dim=5).to("cuda").eval()
    model.load_state_dict(nn_ckpt["model_state_dict"])
    with torch.inference_mode():
        chunks = []
        for start in range(0, len(x), 4096):
            out = model(torch.from_numpy(x[start:start + 4096]).to("cuda"))
            chunks.append(np.column_stack([out[key].cpu().numpy().ravel() for key in TARGETS]))
        nn_pred = np.concatenate(chunks)

    booster = xgb.Booster()
    booster.load_model(XGB_ROOT / "models" / "xgb_common_holes_v2.json")
    xgb_pred = booster.predict(xgb.DMatrix(x)).reshape(len(x), len(TARGETS))

    models = {"ResidualTransportMLP_V3": nn_pred, "XGBoost_one_output_MSE_V2": xgb_pred}
    report = {
        "split": {"strategy": "leave_fixed_holes_out", "holdout_holes": holdout, "n_validation": int(len(indices))},
        "units": {"delta": "percent", "xptar": "rad", "yptar": "rad", "ytar": "cm"},
        "metrics": {name: {"overall": metrics(pred, target, tol, mask, scales), "per_foil": evaluate_by_foil(pred, target, tol, mask, scales, foils)} for name, pred in models.items()},
    }

    for destination in (ROOT / "results" / "common_benchmark.json", XGB_ROOT / "results" / "common_benchmark.json"):
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for name, payload in report["metrics"].items():
        for target_name in TARGETS:
            rows.append({"model": name, "target": target_name, **{metric: payload["overall"][f"{target_name}_{metric}"] for metric in ("center_rmse", "center_mae", "deadzone_rmse", "within_tol")}})
    for destination in (ROOT / "results" / "common_benchmark.csv", XGB_ROOT / "results" / "common_benchmark.csv"):
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for axis, target_name in zip(axes, TARGETS):
        values = [report["metrics"][name]["overall"][f"{target_name}_center_rmse"] for name in models]
        axis.bar(list(models), values, color=["#1565c0", "#c62828"])
        axis.set_title(target_name); axis.set_ylabel("centre RMSE")
        axis.tick_params(axis="x", rotation=20, labelsize=7)
        for i, value in enumerate(values): axis.text(i, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Common held-out-hole validation")
    fig.tight_layout()
    for destination in (ROOT / "results" / "common_benchmark_rmse.png", XGB_ROOT / "results" / "common_benchmark_rmse.png"):
        fig.savefig(destination, dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
