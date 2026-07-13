#!/usr/bin/env python3
"""Analyze Stage-2 ytar distributions and compare NN vs ROOT/HCANA baseline.

This script is tailored to the latest Stage-2 transport fine-tuning workflow.
It loads a `best_finetune.pth` checkpoint plus the labelled Stage-2 CSV, runs
inference, inverse-transforms the network outputs back to physical units, and
produces:

1. A 2x2 distribution comparison figure for all foils and per-foil slices.
2. A dedicated foil-focus figure (default: foil 1) with residual comparison.
3. CSV/JSON summaries containing width/RMSE/within-tolerance metrics.

Baseline/traditional method in this context refers to the existing HCANA/ROOT
reconstruction branch `P_react_z` available in the labelled Stage-2 table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.data.preprocessing import ScalerBundle
from training.data.stage2_root_dataset import Stage2RootDataset
from training.models import build_model_from_config


_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]
_DEFAULT_CHECKPOINT = (
    "checkpoints/stage2_transport_fullroot_25521_general_gpu_clean/best_finetune.pth"
)
_DEFAULT_DATA = "dataset/stage2_25521_labeled.csv"
_DEFAULT_OUTPUT = "outputs/stage2_ytar_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latest Stage-2 ytar distributions")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT, help="Path to stage2 best_finetune.pth")
    parser.add_argument("--data", default=_DEFAULT_DATA, help="Path to stage2 labelled CSV/Parquet/ROOT")
    parser.add_argument("--scaler-bundle", default=None, help="Optional override for scaler bundle path")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT, help="Directory for plots and summaries")
    parser.add_argument("--focus-foil", type=int, default=1, help="Foil index to highlight")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (default: auto)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Inference batch size")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for smoke tests")
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        req = requested.strip().lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_checkpoint_bundle(repo_root: Path, checkpoint_path: Path, scaler_override: str | None, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = dict(cfg.get("model", {}))
    input_dim = int(model_cfg.get("input_dim", 5))
    model = build_model_from_config(model_cfg, input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    scaler_rel = scaler_override or cfg.get("pretrained", {}).get("scaler_bundle_path")
    if not scaler_rel:
        raise RuntimeError("Could not determine scaler bundle path from checkpoint/config.")
    scaler_path = resolve_path(repo_root, scaler_rel)
    scaler_bundle = ScalerBundle.load(str(scaler_path))
    return ckpt, cfg, model, scaler_bundle, scaler_path


def build_dataset(data_path: Path, cfg: dict, scaler_bundle: ScalerBundle, max_events: int | None) -> Stage2RootDataset:
    dcfg = cfg.get("data", {})
    return Stage2RootDataset(
        data_source=str(data_path),
        tree_name=dcfg.get("tree_name", "T"),
        scaler_bundle=scaler_bundle,
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


@torch.no_grad()
def predict_physical(model: torch.nn.Module, dataset: Stage2RootDataset, scaler_bundle: ScalerBundle, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_blocks: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["inputs"].to(device)
        out = model(inputs)
        pred_scaled = torch.cat([out[k] for k in _TARGET_KEYS], dim=1).detach().cpu().numpy()
        pred_blocks.append(pred_scaled)
    pred_scaled_all = np.concatenate(pred_blocks, axis=0).astype(np.float64)
    return scaler_bundle.inverse_transform_Y(pred_scaled_all)


def robust_sigma(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    q16, q84 = np.quantile(arr, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def within_tol(value: np.ndarray, center: np.ndarray, tol: np.ndarray) -> float:
    return float(np.mean(np.abs(value - center) <= tol))


def describe_slice(name: str, ytar_nn: np.ndarray, ytar_root: np.ndarray, center: np.ndarray, tol: np.ndarray) -> Dict[str, Any]:
    nn_res = ytar_nn - center
    root_res = ytar_root - center
    return {
        "slice": name,
        "n_events": int(len(ytar_nn)),
        "center_mean": float(np.mean(center)),
        "center_std": float(np.std(center)),
        "tol_mean": float(np.mean(tol)),
        "tol_std": float(np.std(tol)),
        "nn_mean": float(np.mean(ytar_nn)),
        "nn_std": float(np.std(ytar_nn)),
        "nn_sigma68": robust_sigma(nn_res),
        "nn_rmse_to_center": rmse(ytar_nn, center),
        "nn_mae_to_center": mae(ytar_nn, center),
        "nn_bias_to_center": float(np.mean(nn_res)),
        "nn_within_tol": within_tol(ytar_nn, center, tol),
        "root_mean": float(np.mean(ytar_root)),
        "root_std": float(np.std(ytar_root)),
        "root_sigma68": robust_sigma(root_res),
        "root_rmse_to_center": rmse(ytar_root, center),
        "root_mae_to_center": mae(ytar_root, center),
        "root_bias_to_center": float(np.mean(root_res)),
        "root_within_tol": within_tol(ytar_root, center, tol),
        "rmse_improvement_pct": float(100.0 * (rmse(ytar_root, center) - rmse(ytar_nn, center)) / max(rmse(ytar_root, center), 1e-12)),
        "sigma68_improvement_pct": float(100.0 * (robust_sigma(root_res) - robust_sigma(nn_res)) / max(abs(robust_sigma(root_res)), 1e-12)),
    }


def make_hist(ax, values_a: np.ndarray, values_b: np.ndarray, label_a: str, label_b: str, title: str, center: float | None = None, tol: float | None = None) -> None:
    finite = np.concatenate([values_a[np.isfinite(values_a)], values_b[np.isfinite(values_b)]])
    if finite.size == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
        ax.set_title(title)
        return
    lo, hi = np.quantile(finite, [0.005, 0.995])
    pad = max((hi - lo) * 0.08, 1e-3)
    bins = np.linspace(lo - pad, hi + pad, 80)
    ax.hist(values_b, bins=bins, density=True, alpha=0.45, color="#7f7f7f", label=label_b)
    ax.hist(values_a, bins=bins, density=True, alpha=0.45, color="#1f77b4", label=label_a)
    if center is not None and np.isfinite(center):
        ax.axvline(center, color="#d62728", linestyle="-", linewidth=1.8, label="weak center")
    if tol is not None and np.isfinite(tol):
        ax.axvspan(center - tol, center + tol, color="#ff9896", alpha=0.12, label="tolerance window")
    ax.set_title(title)
    ax.set_xlabel("ytar [cm]")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.15)


def make_residual_hist(ax, res_nn: np.ndarray, res_root: np.ndarray, tol: float | None, title: str) -> None:
    finite = np.concatenate([res_nn[np.isfinite(res_nn)], res_root[np.isfinite(res_root)]])
    lo, hi = np.quantile(finite, [0.005, 0.995])
    pad = max((hi - lo) * 0.08, 1e-3)
    bins = np.linspace(lo - pad, hi + pad, 80)
    ax.hist(res_root, bins=bins, density=True, alpha=0.45, color="#7f7f7f", label="ROOT residual")
    ax.hist(res_nn, bins=bins, density=True, alpha=0.45, color="#1f77b4", label="NN residual")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    if tol is not None and np.isfinite(tol):
        ax.axvspan(-tol, tol, color="#ff9896", alpha=0.12, label="weak tolerance")
    ax.set_title(title)
    ax.set_xlabel("Residual to weak foil center [cm]")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.15)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = resolve_path(repo_root, args.checkpoint)
    data_path = resolve_path(repo_root, args.data)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    ckpt, cfg, model, scaler_bundle, scaler_path = load_checkpoint_bundle(
        repo_root, checkpoint_path, args.scaler_bundle, device
    )
    dataset = build_dataset(data_path, cfg, scaler_bundle, args.max_events)
    pred_phys = predict_physical(model, dataset, scaler_bundle, device, args.batch_size)

    df = dataset.df.reset_index(drop=True).copy()
    df["nn_ytar"] = pred_phys[:, 3]
    df["root_ytar"] = df["P_react_z"].to_numpy(dtype=np.float64)
    df["weak_center_ytar"] = df["weak_foil_ytar_center"].to_numpy(dtype=np.float64)
    df["weak_tol_ytar"] = df["weak_foil_ytar_tol"].to_numpy(dtype=np.float64)

    foil_values = sorted(int(v) for v in df["foil_position"].dropna().unique())
    summaries: list[Dict[str, Any]] = []
    summaries.append(
        describe_slice(
            "all",
            df["nn_ytar"].to_numpy(),
            df["root_ytar"].to_numpy(),
            df["weak_center_ytar"].to_numpy(),
            df["weak_tol_ytar"].to_numpy(),
        )
    )
    for foil in foil_values:
        dff = df.loc[df["foil_position"] == foil]
        summaries.append(
            describe_slice(
                f"foil{foil}",
                dff["nn_ytar"].to_numpy(),
                dff["root_ytar"].to_numpy(),
                dff["weak_center_ytar"].to_numpy(),
                dff["weak_tol_ytar"].to_numpy(),
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "ytar_distribution_summary.csv", index=False)

    focus_foil = int(args.focus_foil)
    focus_df = df.loc[df["foil_position"] == focus_foil].copy()
    if focus_df.empty:
        raise RuntimeError(f"Focus foil {focus_foil} produced an empty slice.")
    focus_center = float(np.median(focus_df["weak_center_ytar"]))
    focus_tol = float(np.median(focus_df["weak_tol_ytar"]))

    # Figure 1: overall + per foil distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    all_center_map = {
        foil: float(np.median(df.loc[df["foil_position"] == foil, "weak_center_ytar"]))
        for foil in foil_values
    }
    all_tol_map = {
        foil: float(np.median(df.loc[df["foil_position"] == foil, "weak_tol_ytar"]))
        for foil in foil_values
    }

    ax0 = axes[0, 0]
    make_hist(
        ax0,
        df["nn_ytar"].to_numpy(),
        df["root_ytar"].to_numpy(),
        "NN ytar",
        "ROOT/HCANA ytar",
        "All foils: ytar distribution",
    )
    for foil in foil_values:
        ax0.axvline(all_center_map[foil], linestyle="--", linewidth=1.0, alpha=0.7, label=f"foil{foil} center" if foil == foil_values[0] else None, color="#d62728")
    ax0.legend(fontsize=9)

    foil_panels = axes.flatten()[1:]
    for ax, foil in zip(foil_panels, foil_values):
        dff = df.loc[df["foil_position"] == foil]
        info = summary_df.loc[summary_df["slice"] == f"foil{foil}"].iloc[0]
        center = float(np.median(dff["weak_center_ytar"]))
        tol = float(np.median(dff["weak_tol_ytar"]))
        make_hist(
            ax,
            dff["nn_ytar"].to_numpy(),
            dff["root_ytar"].to_numpy(),
            "NN ytar",
            "ROOT/HCANA ytar",
            f"foil{foil}: σ68 ROOT={info['root_sigma68']:.3f} cm, NN={info['nn_sigma68']:.3f} cm",
            center=center,
            tol=tol,
        )
        ax.legend(fontsize=8)

    dist_path = output_dir / "ytar_distribution_by_foil.png"
    fig.savefig(dist_path, dpi=180)
    plt.close(fig)

    # Figure 2: focus foil detailed comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    make_hist(
        axes2[0],
        focus_df["nn_ytar"].to_numpy(),
        focus_df["root_ytar"].to_numpy(),
        "NN ytar",
        "ROOT/HCANA ytar",
        f"foil{focus_foil} distribution focus",
        center=focus_center,
        tol=focus_tol,
    )
    focus_info = summary_df.loc[summary_df["slice"] == f"foil{focus_foil}"].iloc[0]
    axes2[0].text(
        0.03,
        0.97,
        (
            f"n={int(focus_info['n_events'])}\n"
            f"RMSE to weak center: ROOT={focus_info['root_rmse_to_center']:.3f} cm, NN={focus_info['nn_rmse_to_center']:.3f} cm\n"
            f"Within tol: ROOT={100*focus_info['root_within_tol']:.1f}%, NN={100*focus_info['nn_within_tol']:.1f}%"
        ),
        transform=axes2[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axes2[0].legend(fontsize=9)

    nn_res = focus_df["nn_ytar"].to_numpy() - focus_df["weak_center_ytar"].to_numpy()
    root_res = focus_df["root_ytar"].to_numpy() - focus_df["weak_center_ytar"].to_numpy()
    make_residual_hist(
        axes2[1],
        nn_res,
        root_res,
        focus_tol,
        f"foil{focus_foil} residual to weak foil center",
    )
    axes2[1].text(
        0.03,
        0.97,
        (
            f"σ68 improvement = {focus_info['sigma68_improvement_pct']:.1f}%\n"
            f"RMSE improvement = {focus_info['rmse_improvement_pct']:.1f}%\n"
            f"bias ROOT={focus_info['root_bias_to_center']:.3f} cm\n"
            f"bias NN={focus_info['nn_bias_to_center']:.3f} cm"
        ),
        transform=axes2[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    axes2[1].legend(fontsize=9)

    focus_path = output_dir / f"ytar_foil{focus_foil}_focus.png"
    fig2.savefig(focus_path, dpi=180)
    plt.close(fig2)

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if ckpt.get("epoch", None) is not None else None,
        "checkpoint_val_loss": float(ckpt.get("val_loss", float("nan"))) if ckpt.get("val_loss", None) is not None else None,
        "scaler_bundle": str(scaler_path),
        "data": str(data_path),
        "device": str(device),
        "dataset_summary": {
            "raw_events": int(dataset.summary.raw_events),
            "kept_events": int(dataset.summary.kept_events),
            "cutflow": dataset.summary.cutflow,
        },
        "focus_foil": focus_foil,
        "summaries": summary_df.to_dict(orient="records"),
        "artifacts": {
            "distribution_plot": str(dist_path),
            "focus_plot": str(focus_path),
            "summary_csv": str(output_dir / "ytar_distribution_summary.csv"),
        },
    }
    save_json(output_dir / "ytar_distribution_summary.json", payload)

    print("Stage-2 ytar analysis complete")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  data       : {data_path}")
    print(f"  output dir : {output_dir}")
    print(f"  kept events: {dataset.summary.kept_events}")
    print(f"  focus foil : {focus_foil}")
    print("\nSummary table:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df[[
            "slice",
            "n_events",
            "root_sigma68",
            "nn_sigma68",
            "root_rmse_to_center",
            "nn_rmse_to_center",
            "root_within_tol",
            "nn_within_tol",
            "rmse_improvement_pct",
            "sigma68_improvement_pct",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
