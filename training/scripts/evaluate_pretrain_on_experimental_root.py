#!/usr/bin/env python3
"""Validate a pretrained transport model on experimental SHMS ROOT data.

Unlike SIMC evaluation, experimental replay ROOT files do not contain MC truth.
This script therefore compares the pretrained network predictions against the
existing HCANA/ROOT reconstruction as a *baseline reference*, not as truth.

Outputs
-------
- metrics_vs_root.json
- metrics_vs_root.csv
- prediction_summary.csv
- cutflow_summary.csv
- parity_vs_root.png
- residual_vs_root.png
- output_distributions.png
- diff_vs_root_scatter.png
- diff_vs_features.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from training.data.preprocessing import ScalerBundle
from training.models import build_model_from_config

plt.style.use("seaborn-v0_8-whitegrid")

INPUT_BRANCHES = ["P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp"]
TARGET_BRANCHES = {
    "delta": "P_gtr_dp",
    "xptar": "P_gtr_th",
    "yptar": "P_gtr_ph",
    "ytar": "P_react_z",
}
FRY_ADC_BRANCHES = ["P_rb_raster_fryaRawAdc", "P_rb_raster_frybRawAdc"]
PID_BRANCHES = ["P_ngcer_npeSum", "P_hgcer_npeSum", "P_aero_npeSum", "P_cal_etottracknorm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pretrained SHMS model on experimental ROOT data")
    parser.add_argument("--checkpoint", required=True, help="Path to best_pretrain.pth")
    parser.add_argument("--root-file", required=True, help="Experimental ROOT file path")
    parser.add_argument("--tree-name", default="T", help="ROOT tree name")
    parser.add_argument("--scaler-bundle", default=None, help="Path to scaler_bundle.json (default: next to checkpoint)")
    parser.add_argument("--device", default=None, help="cuda/cpu")
    parser.add_argument("--batch-size", type=int, default=8192, help="Inference batch size")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
    parser.add_argument("--fry-mode", choices=["adc-normalized", "mc-mean"], default="adc-normalized", help="How to construct the 5th fry feature from experimental raster information")
    parser.add_argument("--disable-quality-cut", action="store_true", help="Keep all finite events; otherwise apply mild reco sanity cuts")
    parser.add_argument("--disable-pid-cut", action="store_true", help="Disable default PID cuts")
    parser.add_argument("--ngcer-min", type=float, default=2.0, help="Minimum P_ngcer_npeSum for PID selection")
    parser.add_argument("--hgcer-min", type=float, default=0.5, help="Minimum P_hgcer_npeSum for PID selection")
    parser.add_argument("--aero-min", type=float, default=None, help="Optional minimum P_aero_npeSum for PID selection")
    parser.add_argument("--cal-etot-min", type=float, default=0.6, help="Minimum P_cal_etottracknorm for PID selection")
    parser.add_argument("--cal-etot-max", type=float, default=1.8, help="Maximum P_cal_etottracknorm for PID selection")
    parser.add_argument("--dp-min", type=float, default=-25.0, help="Minimum P_gtr_dp for physics selection")
    parser.add_argument("--dp-max", type=float, default=22.0, help="Maximum P_gtr_dp for physics selection")
    parser.add_argument("--xptar-abs-max", type=float, default=0.08, help="Maximum absolute P_gtr_th for physics selection")
    parser.add_argument("--yptar-abs-max", type=float, default=0.06, help="Maximum absolute P_gtr_ph for physics selection")
    parser.add_argument("--ytar-abs-max", type=float, default=120.0, help="Maximum absolute P_react_z for physics selection")
    parser.add_argument("--output-dir", default=None, help="Where to save metrics and plots")
    return parser.parse_args()


def load_checkpoint_and_scaler(checkpoint_path: Path, scaler_path: Path | None, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if scaler_path is None:
        scaler_path = checkpoint_path.parent / "scaler_bundle.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler bundle not found: {scaler_path}")
    scaler_bundle = ScalerBundle.load(str(scaler_path))
    model_cfg = dict(ckpt.get("config", {}).get("model", {}))
    data_cfg = dict(ckpt.get("config", {}).get("data", {}))
    model = build_model_from_config(model_cfg, input_dim=len(scaler_bundle.input_features))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return ckpt, scaler_bundle, model, data_cfg


def build_fry_proxy(df: pd.DataFrame, scaler_bundle: ScalerBundle, mode: str) -> np.ndarray:
    mc_fry_mean = float(scaler_bundle.scaler_X.mean_[4])
    mc_fry_scale = float(scaler_bundle.scaler_X.scale_[4])
    if mode == "mc-mean":
        return np.full(len(df), mc_fry_mean, dtype=np.float32)

    avg_adc = df[FRY_ADC_BRANCHES].mean(axis=1).to_numpy(dtype=np.float64)
    adc_mean = float(np.mean(avg_adc))
    adc_std = float(np.std(avg_adc))
    if adc_std < 1e-12:
        return np.full(len(df), mc_fry_mean, dtype=np.float32)
    z = (avg_adc - adc_mean) / adc_std
    fry_proxy = mc_fry_mean + z * mc_fry_scale
    return fry_proxy.astype(np.float32)


def finite_mask(df: pd.DataFrame) -> pd.Series:
    required = INPUT_BRANCHES + list(TARGET_BRANCHES.values()) + FRY_ADC_BRANCHES + PID_BRANCHES
    return np.isfinite(df[required]).all(axis=1)


def load_experimental_dataframe(root_file: Path, tree_name: str, max_events: int | None) -> pd.DataFrame:
    import uproot

    branches = [*INPUT_BRANCHES, *TARGET_BRANCHES.values(), *FRY_ADC_BRANCHES, *PID_BRANCHES]
    with uproot.open(root_file) as f:
        tree = f[tree_name]
        df = tree.arrays(branches, library="pd")
    if max_events is not None:
        df = df.iloc[:max_events].copy()
    return df


def apply_physics_cut(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    mask = df["P_gtr_dp"].between(args.dp_min, args.dp_max)
    mask &= df["P_gtr_th"].abs() <= args.xptar_abs_max
    mask &= df["P_gtr_ph"].abs() <= args.yptar_abs_max
    mask &= df["P_react_z"].abs() <= args.ytar_abs_max
    return mask


def apply_pid_cut(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    mask = df["P_ngcer_npeSum"] >= args.ngcer_min
    mask &= df["P_hgcer_npeSum"] >= args.hgcer_min
    mask &= df["P_cal_etottracknorm"].between(args.cal_etot_min, args.cal_etot_max)
    if args.aero_min is not None:
        mask &= df["P_aero_npeSum"] >= args.aero_min
    return mask


def build_cutflow(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.Series, pd.DataFrame]:
    base = pd.Series(True, index=df.index)
    finite = finite_mask(df)
    physics = apply_physics_cut(df, args) if not args.disable_quality_cut else pd.Series(True, index=df.index)
    pid = apply_pid_cut(df, args) if not args.disable_pid_cut else pd.Series(True, index=df.index)
    final_mask = base & finite & physics & pid

    rows = []
    current = base
    for name, step_mask in [
        ("raw", base),
        ("finite", finite),
        ("physics", physics),
        ("pid", pid),
        ("final", final_mask),
    ]:
        if name == "raw":
            current = base
        elif name == "final":
            current = final_mask
        else:
            current = current & step_mask
        rows.append(
            {
                "step": name,
                "events": int(current.sum()),
                "fraction_of_raw": float(current.mean()),
            }
        )
    return final_mask, pd.DataFrame(rows)


@torch.no_grad()
def predict(model: torch.nn.Module, x_scaled: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    xs = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    chunks: list[np.ndarray] = []
    for start in range(0, len(xs), batch_size):
        xb = xs[start:start + batch_size]
        out = model(xb)
        total = np.column_stack([out[key].squeeze(-1).detach().cpu().numpy() for key in ["delta", "xptar", "yptar", "ytar"]])
        chunks.append(total)
    return np.concatenate(chunks, axis=0)


def compare_to_root(pred: np.ndarray, root: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, target in enumerate(target_names):
        diff = pred[:, i] - root[:, i]
        rows.append(
            {
                "target": target,
                "nn_mean": float(np.mean(pred[:, i])),
                "root_mean": float(np.mean(root[:, i])),
                "bias_nn_minus_root": float(np.mean(diff)),
                "std_nn_minus_root": float(np.std(diff)),
                "rmse_nn_vs_root": float(np.sqrt(np.mean(diff ** 2))),
                "mae_nn_vs_root": float(np.mean(np.abs(diff))),
                "corr_nn_root": float(np.corrcoef(pred[:, i], root[:, i])[0, 1]) if np.std(pred[:, i]) > 1e-12 and np.std(root[:, i]) > 1e-12 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def make_prediction_summary(pred: np.ndarray, root: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, target in enumerate(target_names):
        for source_name, values in [("nn", pred[:, i]), ("root", root[:, i])]:
            rows.append(
                {
                    "target": target,
                    "source": source_name,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "q01": float(np.quantile(values, 0.01)),
                    "q50": float(np.quantile(values, 0.50)),
                    "q99": float(np.quantile(values, 0.99)),
                }
            )
    return pd.DataFrame(rows)


def make_diff_summary(pred: np.ndarray, root: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, target in enumerate(target_names):
        diff = pred[:, i] - root[:, i]
        rows.append(
            {
                "target": target,
                "diff_mean": float(np.mean(diff)),
                "diff_std": float(np.std(diff)),
                "diff_median": float(np.median(diff)),
                "diff_q01": float(np.quantile(diff, 0.01)),
                "diff_q99": float(np.quantile(diff, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def plot_parity(pred: np.ndarray, root: np.ndarray, target_names: list[str], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 4))
    axes_list = list(axes.ravel()) if isinstance(axes, np.ndarray) else [axes]
    for i, target in enumerate(target_names):
        ax = axes_list[i]
        x = root[:, i]
        y = pred[:, i]
        ax.scatter(x, y, s=4, alpha=0.15)
        lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
        hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"NN vs ROOT: {target}")
        ax.set_xlabel("ROOT reco")
        ax.set_ylabel("NN pred")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals(pred: np.ndarray, root: np.ndarray, target_names: list[str], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 4))
    axes_list = list(axes.ravel()) if isinstance(axes, np.ndarray) else [axes]
    for i, target in enumerate(target_names):
        ax = axes_list[i]
        diff = pred[:, i] - root[:, i]
        ax.hist(diff, bins=80, alpha=0.7)
        ax.axvline(0.0, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"Residual NN-ROOT: {target}")
        ax.set_xlabel("NN - ROOT")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_distributions(pred: np.ndarray, root: np.ndarray, target_names: list[str], save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 4))
    axes_list = list(axes.ravel()) if isinstance(axes, np.ndarray) else [axes]
    for i, target in enumerate(target_names):
        ax = axes_list[i]
        ax.hist(root[:, i], bins=100, histtype="step", linewidth=1.8, label="ROOT reco", density=True)
        ax.hist(pred[:, i], bins=100, alpha=0.45, label="NN pred", density=True)
        ax.set_title(f"Distribution: {target}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_diff_vs_root(pred: np.ndarray, root: np.ndarray, target_names: list[str], save_path: Path) -> None:
    fig, axes = plt.subplots(2, len(target_names), figsize=(5 * len(target_names), 8))
    for i, target in enumerate(target_names):
        diff = pred[:, i] - root[:, i]
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
        ax_top.scatter(root[:, i], diff, s=3, alpha=0.12)
        ax_top.axhline(0.0, color="r", linestyle="--", linewidth=1)
        ax_top.set_title(f"Δ vs ROOT: {target}")
        ax_top.set_xlabel("ROOT reco")
        ax_top.set_ylabel("NN - ROOT")
        ax_bot.hexbin(root[:, i], diff, gridsize=45, mincnt=1, cmap="viridis")
        ax_bot.axhline(0.0, color="r", linestyle="--", linewidth=1)
        ax_bot.set_xlabel("ROOT reco")
        ax_bot.set_ylabel("NN - ROOT")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_diff_vs_features(pred: np.ndarray, root: np.ndarray, features: np.ndarray, feature_names: list[str], target_names: list[str], save_path: Path) -> None:
    fig, axes = plt.subplots(len(target_names), len(feature_names), figsize=(3.2 * len(feature_names), 3.0 * len(target_names)), squeeze=False)
    for i, target in enumerate(target_names):
        diff = pred[:, i] - root[:, i]
        for j, feature_name in enumerate(feature_names):
            ax = axes[i, j]
            ax.scatter(features[:, j], diff, s=2, alpha=0.08)
            ax.axhline(0.0, color="r", linestyle="--", linewidth=0.8)
            if i == 0:
                ax.set_title(feature_name)
            if j == 0:
                ax.set_ylabel(f"{target}\nNN - ROOT")
            if i == len(target_names) - 1:
                ax.set_xlabel(feature_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint).resolve()
    root_path = Path(args.root_file).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint_path.parent / f"eval_experimental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt, scaler_bundle, model, data_cfg = load_checkpoint_and_scaler(
        checkpoint_path,
        Path(args.scaler_bundle).resolve() if args.scaler_bundle else None,
        device,
    )

    df = load_experimental_dataframe(root_path, args.tree_name, args.max_events)
    raw_events = len(df)
    mask, cutflow_df = build_cutflow(df, args)
    df = df.loc[mask].reset_index(drop=True)

    fry = build_fry_proxy(df, scaler_bundle, mode=args.fry_mode)
    x_raw = np.column_stack([
        df["P_dc_x_fp"].to_numpy(dtype=np.float32),
        df["P_dc_y_fp"].to_numpy(dtype=np.float32),
        df["P_dc_xp_fp"].to_numpy(dtype=np.float32),
        df["P_dc_yp_fp"].to_numpy(dtype=np.float32),
        fry,
    ])
    feature_names = ["x_fp", "y_fp", "xp_fp", "yp_fp", "fry"]
    x_scaled = scaler_bundle.transform_X(x_raw).astype(np.float32)

    pred_scaled = predict(model, x_scaled, device=device, batch_size=args.batch_size)
    pred = scaler_bundle.inverse_transform_Y(pred_scaled)
    root = np.column_stack([df[branch].to_numpy(dtype=np.float32) for branch in TARGET_BRANCHES.values()])

    target_names = list(TARGET_BRANCHES.keys())
    metrics_df = compare_to_root(pred, root, target_names)
    summary_df = make_prediction_summary(pred, root, target_names)
    diff_summary_df = make_diff_summary(pred, root, target_names)

    metrics_csv = output_dir / "metrics_vs_root.csv"
    metrics_json = output_dir / "metrics_vs_root.json"
    summary_csv = output_dir / "prediction_summary.csv"
    cutflow_csv = output_dir / "cutflow_summary.csv"
    diff_summary_csv = output_dir / "diff_summary.csv"
    parity_png = output_dir / "parity_vs_root.png"
    residual_png = output_dir / "residual_vs_root.png"
    dist_png = output_dir / "output_distributions.png"
    diff_vs_root_png = output_dir / "diff_vs_root_scatter.png"
    diff_vs_features_png = output_dir / "diff_vs_features.png"

    metrics_df.to_csv(metrics_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    cutflow_df.to_csv(cutflow_csv, index=False)
    diff_summary_df.to_csv(diff_summary_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "root_file": str(root_path),
                "tree_name": args.tree_name,
                "device": str(device),
                "fry_mode": args.fry_mode,
                "quality_cut": not args.disable_quality_cut,
                "pid_cut": not args.disable_pid_cut,
                "raw_events": raw_events,
                "used_events": int(len(df)),
                "input_features": scaler_bundle.input_features,
                "target_branches": TARGET_BRANCHES,
                "cut_parameters": {
                    "ngcer_min": args.ngcer_min,
                    "hgcer_min": args.hgcer_min,
                    "aero_min": args.aero_min,
                    "cal_etot_min": args.cal_etot_min,
                    "cal_etot_max": args.cal_etot_max,
                    "dp_min": args.dp_min,
                    "dp_max": args.dp_max,
                    "xptar_abs_max": args.xptar_abs_max,
                    "yptar_abs_max": args.yptar_abs_max,
                    "ytar_abs_max": args.ytar_abs_max,
                },
                "cutflow": cutflow_df.to_dict(orient="records"),
                "metrics_vs_root": metrics_df.to_dict(orient="records"),
                "diff_summary": diff_summary_df.to_dict(orient="records"),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    plot_parity(pred, root, target_names, parity_png)
    plot_residuals(pred, root, target_names, residual_png)
    plot_distributions(pred, root, target_names, dist_png)
    plot_diff_vs_root(pred, root, target_names, diff_vs_root_png)
    plot_diff_vs_features(pred, root, x_raw, feature_names, target_names, diff_vs_features_png)

    print(f"Experimental ROOT evaluation directory: {output_dir}")
    print(f"Events: raw={raw_events}, used={len(df)}")
    print("Cutflow:")
    print(cutflow_df.to_string(index=False))
    print(metrics_df.to_string(index=False))
    print("Saved:")
    for path in [metrics_csv, metrics_json, summary_csv, cutflow_csv, diff_summary_csv, parity_png, residual_png, dist_png, diff_vs_root_png, diff_vs_features_png]:
        print("-", path)


if __name__ == "__main__":
    main()
