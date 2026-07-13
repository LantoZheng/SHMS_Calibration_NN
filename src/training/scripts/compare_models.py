#!/usr/bin/env python3
"""
Compare GUI-optimized model vs V3 baseline model on the same GUI-labeled data.
Generates scatter plots, residual histograms, and per-foil bar charts.
"""

from __future__ import annotations

import json, os, sys, time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from training.data.stage2_root_dataset import Stage2RootDataset
from training.data.preprocessing import ScalerBundle
from training.models import build_model_from_config
from training.losses import Stage2WeakLabelLoss

_TARGETS = ["ytar", "xptar", "yptar", "delta"]
_TARGET_LABELS = {
    "ytar": r"$y_{\rm tar}$ (cm)",
    "xptar": r"$x'_{\rm tar}$ (rad)",
    "yptar": r"$y'_{\rm tar}$ (rad)",
    "delta": r"$\delta$ (%)",
}
_MODEL_PATHS = {
    "GUI v2 (fixed config)": str(_REPO / "checkpoints/stage2_gui_v2/best_finetune.pth"),
    "V3 Baseline (NN-relabel)": str(_REPO / "checkpoints/stage2_transport_fullroot_25521_mainline_centerout_nnrelabel_v3/best_finetune.pth"),
    "V3 Iter1 (GUI preserve)": str(_REPO / "outputs/iterative_gui_v3_preserved/checkpoints/iter1/best_finetune.pth"),
}
_SCALER_PATH = str(_REPO / "checkpoints/pretrain_25521_fry_cuda_5d_notebooklike/scaler_bundle.json")
_DATA_PATH = str(_REPO.parent / "stage2_soc_gui_labeled.csv")  # relative to SHMS_Calibration_NN
_CONFIG_PATH = str(_REPO / "training/configs/v3_gui_optimized.yaml")
_OUTPUT_DIR = _REPO / "plots/model_comparison"


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})
    model = build_model_from_config(mcfg, input_dim=mcfg.get("input_dim", 5))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def load_data(csv_path: str, scaler_path: str, max_events: int | None = None) -> tuple:
    scaler = ScalerBundle.load(scaler_path)
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg.get("data", {})
    ds = Stage2RootDataset(
        data_source=csv_path,
        tree_name=dcfg.get("tree_name", "T"),
        scaler_bundle=scaler,
        feature_schema=dcfg.get("feature_schema", ["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"]),
        branch_map=dcfg.get("branch_map", {}),
        label_map=dcfg.get("label_map", {}),
        metadata_cols=dcfg.get("metadata_cols", {}),
        weight_col=dcfg.get("weight_col", None),
        fry_mode=dcfg.get("fry_mode", "direct_or_proxy"),
        direct_fry_branch=dcfg.get("direct_fry_branch", None),
        fry_proxy_branches=dcfg.get("fry_proxy_branches", []),
        cuts=dcfg.get("cuts", {}),
        max_events=max_events,
    )
    return ds, scaler


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=4096) -> dict[str, np.ndarray]:
    """Run inference and return unnormalized predictions + targets."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = {k: [] for k in _TARGETS}
    all_targets = {k: [] for k in _TARGETS}
    all_foils = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        preds = model(inputs)

        for k in _TARGETS:
            all_preds[k].append(preds[k].cpu().numpy().ravel())
            raw_k = batch.get("targets", {}).get(k, None)
            if raw_k is not None:
                all_targets[k].append(raw_k.cpu().numpy().ravel() if isinstance(raw_k, torch.Tensor) else raw_k)

        meta = batch.get("metadata", {})
        if "foil_position" in meta:
            all_foils.append(np.asarray(meta["foil_position"]).ravel())

    result = {}
    for k in _TARGETS:
        result[f"{k}_pred"] = np.concatenate(all_preds[k]) if all_preds[k] else np.array([])
        result[f"{k}_true"] = np.concatenate(all_targets[k]) if all_targets[k] else np.array([])
    result["foil"] = np.concatenate(all_foils) if all_foils else np.array([])
    return result


def compute_metrics(results: dict) -> dict:
    metrics = {}
    for k in _TARGETS:
        pred = results[f"{k}_pred"]
        true = results[f"{k}_true"]
        if len(pred) == 0 or len(true) == 0:
            continue
        mask = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[mask], true[mask]
        if len(pred) == 0:
            continue
        residual = pred - true
        metrics[f"{k}_rmse"] = float(np.sqrt(np.mean(residual ** 2)))
        metrics[f"{k}_mae"] = float(np.mean(np.abs(residual)))
        metrics[f"{k}_bias"] = float(np.mean(residual))
        metrics[f"{k}_std"] = float(np.std(residual))
    return metrics


def compute_per_foil_metrics(results: dict) -> dict[str, dict]:
    foils = results.get("foil", np.array([]))
    per_foil = {}
    for foil_val in sorted(np.unique(foils)):
        foil_key = str(int(foil_val))
        mask = foils == foil_val
        if mask.sum() < 10:
            continue
        foil_results = {}
        for k in _TARGETS:
            foil_results[f"{k}_pred"] = results[f"{k}_pred"][mask]
            foil_results[f"{k}_true"] = results[f"{k}_true"][mask]
        foil_results["foil"] = foils[mask]
        per_foil[foil_key] = compute_metrics(foil_results)
    return per_foil


def plot_scatter_comparison(
    all_results: dict[str, dict],
    target: str,
    ax: plt.Axes,
    max_points: int = 8000,
):
    """Scatter: predicted vs true for each model, one subplot column."""
    colors = {"GUI v2 (fixed config)": "#2196F3", "V3 Baseline (NN-relabel)": "#FF9800", "V3 Iter1 (GUI preserve)": "#4CAF50"}
    markers = {"GUI v2 (fixed config)": "o", "V3 Baseline (NN-relabel)": "s", "V3 Iter1 (GUI preserve)": "D"}

    for name, results in all_results.items():
        pred = results.get(f"{target}_pred", np.array([]))
        true = results.get(f"{target}_true", np.array([]))
        if len(pred) == 0:
            continue
        mask = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[mask], true[mask]
        if len(pred) > max_points:
            idx = np.random.default_rng(42).choice(len(pred), max_points, replace=False)
            pred, true = pred[idx], true[idx]

        ax.scatter(true, pred, c=colors.get(name, "gray"), marker=markers.get(name, "o"),
                   s=1, alpha=0.3, label=name, rasterized=True)

    # y=x line
    lims = [np.inf, -np.inf]
    for results in all_results.values():
        t = results.get(f"{target}_true", np.array([]))
        p = results.get(f"{target}_pred", np.array([]))
        if len(t) > 0:
            m = np.isfinite(t) & np.isfinite(p)
            if m.sum() > 0:
                lims[0] = min(lims[0], t[m].min(), p[m].min())
                lims[1] = max(lims[1], t[m].max(), p[m].max())
    if np.isfinite(lims[0]):
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xlabel(f"True {_TARGET_LABELS[target]}")
    ax.set_ylabel(f"Predicted {_TARGET_LABELS[target]}")
    ax.legend(loc="upper left", fontsize=6, markerscale=4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_residual_histograms(
    all_results: dict[str, dict],
    target: str,
    ax: plt.Axes,
):
    """Overlaid residual histograms."""
    colors_hist = {"GUI v2 (fixed config)": "#2196F3", "V3 Baseline (NN-relabel)": "#FF9800", "V3 Iter1 (GUI preserve)": "#4CAF50"}
    for name, results in all_results.items():
        pred = results.get(f"{target}_pred", np.array([]))
        true = results.get(f"{target}_true", np.array([]))
        if len(pred) == 0:
            continue
        mask = np.isfinite(pred) & np.isfinite(true)
        residual = pred[mask] - true[mask]
        if len(residual) == 0:
            continue
        rmse = np.sqrt(np.mean(residual ** 2))
        ax.hist(residual, bins=80, alpha=0.4, density=True, color=colors_hist.get(name, "gray"),
                label=f"{name} (RMSE={rmse:.4f})")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Residual {_TARGET_LABELS[target]}")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_per_foil_barchart(
    all_per_foil: dict[str, dict[str, dict]],
    target: str,
    ax: plt.Axes,
):
    """Grouped bar chart: RMSE per foil per model."""
    foil_keys = sorted(set().union(*[set(d.keys()) for d in all_per_foil.values()]))
    n_foils = len(foil_keys)
    n_models = len(all_per_foil)
    width = 0.7 / n_models
    x = np.arange(n_foils)
    colors_bar = {"GUI v2 (fixed config)": "#2196F3", "V3 Baseline (NN-relabel)": "#FF9800", "V3 Iter1 (GUI preserve)": "#4CAF50"}

    for i, (name, per_foil) in enumerate(all_per_foil.items()):
        values = [per_foil.get(fk, {}).get(f"{target}_rmse", np.nan) for fk in foil_keys]
        ax.bar(x + i * width - (n_models - 1) * width / 2, values, width, label=name,
               color=colors_bar.get(name, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Foil {fk}" for fk in foil_keys])
    ax.set_ylabel(f"{_TARGET_LABELS[target]} RMSE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def main():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once
    print("Loading dataset...")
    dataset, _ = load_data(_DATA_PATH, _SCALER_PATH)

    # Run inference for each model
    all_results = {}
    all_metrics = {}
    all_per_foil = {}

    for name, ckpt_path in _MODEL_PATHS.items():
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {name}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        model, cfg = load_model(ckpt_path, device)
        t0 = time.time()
        results = run_inference(model, dataset, device)
        elapsed = time.time() - t0
        metrics = compute_metrics(results)
        per_foil = compute_per_foil_metrics(results)
        all_results[name] = results
        all_metrics[name] = metrics
        all_per_foil[name] = per_foil

        print(f"  Inference: {elapsed:.1f}s")
        for k in _TARGETS:
            rmse = metrics.get(f"{k}_rmse", float("nan"))
            mae = metrics.get(f"{k}_mae", float("nan"))
            bias = metrics.get(f"{k}_bias", float("nan"))
            print(f"  {k:>6s}: RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:.4f}")

    # ---- PLOT: 4x3 scatter grid (rows=targets, cols=residual / scatter / per-foil) ----
    n_targets = len(_TARGETS)
    fig, axes = plt.subplots(n_targets, 3, figsize=(16, 4.5 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)

    for i, target in enumerate(_TARGETS):
        plot_scatter_comparison(all_results, target, axes[i, 0])
        plot_residual_histograms(all_results, target, axes[i, 1])
        plot_per_foil_barchart(all_per_foil, target, axes[i, 2])

    fig.suptitle("Model Comparison: GUI Optimized vs V3 Baseline", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "comparison_full.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {_OUTPUT_DIR / 'comparison_full.png'}")

    # ---- PLOT: Summary bar chart of RMSE across models ----
    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(all_metrics)
    width = 0.7 / n_models
    x = np.arange(len(_TARGETS))
    colors_sum = {"GUI v2 (fixed config)": "#2196F3", "V3 Baseline (NN-relabel)": "#FF9800", "V3 Iter1 (GUI preserve)": "#4CAF50"}
    for i, (name, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(f"{t}_rmse", np.nan) for t in _TARGETS]
        ax.bar(x + i * width - (n_models - 1) * width / 2, values, width, label=name,
               color=colors_sum.get(name, "gray"), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([_TARGET_LABELS[t] for t in _TARGETS])
    ax.set_ylabel("RMSE")
    ax.set_title("Per-Target RMSE Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(_OUTPUT_DIR / "comparison_rmse_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {_OUTPUT_DIR / 'comparison_rmse_summary.png'}")

    # ---- Save metrics JSON ----
    summary = {
        "overall": all_metrics,
        "per_foil": {name: {fk: {kk: vv for kk, vv in mv.items()} for fk, mv in pfm.items()}
                      for name, pfm in all_per_foil.items()},
    }
    with open(_OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {_OUTPUT_DIR / 'metrics.json'}")

    # ---- Print summary table ----
    print(f"\n{'='*80}")
    print(f"{'Target':<8} {'Metric':<8}", end="")
    for name in all_metrics:
        print(f" {name:<28}", end="")
    print()
    print("-" * 80)
    for target in _TARGETS:
        for metric in ["rmse", "mae", "bias"]:
            print(f"{target:<8} {metric:<8}", end="")
            for name in all_metrics:
                val = all_metrics[name].get(f"{target}_{metric}", float("nan"))
                print(f" {val:<28.4f}", end="")
            print()


if __name__ == "__main__":
    main()
