#!/usr/bin/env python3
"""
xptar / yptar 1D histogram comparison (per foil) — same style as ztar plot.

KEY INSIGHT: The NN's "xptar"/"yptar" outputs are trained on
  weak_hole_xptar_center = candidate_sieve_x_cm / 253.0
  weak_hole_yptar_center = candidate_sieve_y_cm / 253.0

These are sieve-plane proxy angles.  The honest comparison is in the
SIEVE PLANE, where both NN and ROOT predictions are compared to the
mechanical hole positions — no reliance on HCANA P_gtr_x/y for NN.

  NN   sieve_x = project_to_sieve_full(x_tar, y_tar, nn_xptar, nn_yptar, nn_delta)  [full δ formula]
  ROOT sieve_x = project_to_sieve_full(P_gtr_x, P_gtr_y, P_gtr_th, P_gtr_ph, P_gtr_dp)  [full formula]

Full SHMS sieve projection (not the old linear θ×253 / φ×253):
  sieve_x = x_tar + θ × 253                           [cm]
  sieve_y = (-0.019·δ + 0.00019·δ² + 213·φ + y_tar)
          + 40 × (-0.00052·δ + 0.0000052·δ² + φ)      [cm]
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path

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
_FOIL_ORDER = [0, 1, 2]
_SIEVE_DISTANCE_CM = 253.0


def project_to_sieve_full(x_tar, y_tar, th, ph, dp):
    """Exact SHMS project_to_sieve with default TargetProjectionConfig."""
    sieve_x = x_tar + th * 253.0
    sieve_y = (
        -0.019 * dp + 0.00019 * dp**2 + 213.0 * ph + y_tar
    ) + 40.0 * (
        -0.00052 * dp + 0.0000052 * dp**2 + ph
    )
    return sieve_x, sieve_y


def robust_sigma(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return float("nan")
    q16, q84 = np.quantile(arr, [0.16, 0.84])
    return float(0.5 * (q84 - q16))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--scaler-bundle", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def resolve_path(repo_root, rel):
    p = Path(rel)
    return p if p.is_absolute() else (repo_root / rel).resolve()


def resolve_device(device_str):
    if device_str: return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_bundle(repo_root, checkpoint_path, scaler_override, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = dict(cfg.get("model", {}))
    model = build_model_from_config(model_cfg, input_dim=model_cfg.get("input_dim", 5))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    scaler_rel = scaler_override or cfg["pretrained"].get("scaler_bundle_path")
    scaler = ScalerBundle.load(str(resolve_path(repo_root, scaler_rel)))
    return ckpt, cfg, model, scaler


def build_dataset_from_config(repo_root, data_path, cfg, scaler):
    dcfg = cfg.get("data", {})
    return Stage2RootDataset(
        data_source=str(data_path), tree_name=dcfg.get("tree_name", "T"),
        scaler_bundle=scaler,
        feature_schema=dcfg.get("feature_schema", ["x_fp","y_fp","xp_fp","yp_fp","fry"]),
        branch_map=dcfg.get("branch_map", {}),
        label_map=dcfg.get("label_map", {}),
        metadata_cols=dcfg.get("metadata_cols", {}),
        weight_col=dcfg.get("weight_col"),
        fry_mode=dcfg.get("fry_mode", "direct_or_proxy"),
        direct_fry_branch=dcfg.get("direct_fry_branch"),
        fry_proxy_branches=dcfg.get("fry_proxy_branches", []),
        cuts=dcfg.get("cuts", {}),
    )


@torch.no_grad()
def predict_physical(model, dataset, scaler, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    blocks = []
    for batch in loader:
        out = model(batch["inputs"].to(device))
        blocks.append(torch.cat([out[k] for k in _TARGET_KEYS], dim=1).cpu().numpy())
    return scaler.inverse_transform_Y(np.concatenate(blocks, axis=0).astype(np.float64))


def _plot_sieve_comparison(plot_df, output_path, axis):
    """2×2 1D histogram: NN sieve vs ROOT sieve vs mechanical hole centers."""
    nn_col = f"nn_sieve_{axis}"
    root_col = f"root_sieve_{axis}"
    design_col = f"design_sieve_{axis}"

    foil_values = [f for f in _FOIL_ORDER if f in set(plot_df["foil_position"].dropna().astype(int))]
    if len(foil_values) != 3:
        foil_values = sorted(int(v) for v in plot_df["foil_position"].dropna().unique())

    def _stats(dff, name):
        nn = dff[nn_col].to_numpy(dtype=np.float64)
        root = dff[root_col].to_numpy(dtype=np.float64)
        d = dff[design_col].to_numpy(dtype=np.float64)
        return {
            "slice": name, "n_events": len(dff),
            "nn_rmse_to_design": rmse(nn, d),
            "root_rmse_to_design": rmse(root, d),
            "nn_sigma68": robust_sigma(nn - d),
            "root_sigma68": robust_sigma(root - d),
            "nn_rmse_to_root": rmse(nn, root),
            "nn_sigma68_to_root": robust_sigma(nn - root),
            "nn_bias_to_root": float(np.mean(nn - root)),
            "nn_mean": float(np.mean(nn)), "root_mean": float(np.mean(root)),
        }

    rows = [_stats(plot_df, "all")]
    for f in foil_values:
        rows.append(_stats(plot_df.loc[plot_df["foil_position"] == f], f"foil{f}"))
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    slices_data = [("all", plot_df)] + [
        (f"foil{f}", plot_df.loc[plot_df["foil_position"] == f].copy()) for f in foil_values
    ]

    label_map = {"x": "sieve X", "y": "sieve Y"}

    for ax, (name, dff) in zip(axes.flatten(), slices_data):
        nn = dff[nn_col].to_numpy(dtype=np.float64)
        root = dff[root_col].to_numpy(dtype=np.float64)
        design = dff[design_col].to_numpy(dtype=np.float64)

        finite = np.concatenate([nn[np.isfinite(nn)], root[np.isfinite(root)]])
        lo, hi = np.quantile(finite, [0.005, 0.995])
        pad = max((hi - lo) * 0.08, 0.5)
        bins = np.linspace(lo - pad, hi + pad, 80)

        ax.hist(root, bins=bins, density=True, alpha=0.45, color="#7f7f7f",
                label=f"ROOT/HCANA {label_map[axis]}")
        ax.hist(nn, bins=bins, density=True, alpha=0.45, color="#1f77b4",
                label=f"NN {label_map[axis]}")

        design_unique = sorted({round(float(v), 6) for v in design if np.isfinite(v)})
        for idx, c in enumerate(design_unique):
            ax.axvline(c, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.75,
                       label="hole center" if idx == 0 else None)

        row = summary[summary["slice"] == name].iloc[0]
        ax.set_title(
            f"{name}: NN vs ROOT RMSE={row['nn_rmse_to_root']:.3f} cm, "
            f"σ68={row['nn_sigma68_to_root']:.3f} cm\n"
            f"NN bias to ROOT={row['nn_bias_to_root']:.3f} cm"
        )
        ax.set_xlabel(f"{label_map[axis]} [cm]")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)

    fig.suptitle(f"Stage-2 sieve-{axis} comparison: NN vs ROOT vs mechanical design", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    ckpt, cfg, model, scaler = load_checkpoint_bundle(
        repo_root, resolve_path(repo_root, args.checkpoint), args.scaler_bundle, device)
    dataset = build_dataset_from_config(repo_root, resolve_path(repo_root, args.data), cfg, scaler)
    pred_phys = predict_physical(model, dataset, scaler, device, args.batch_size)
    print(f"Inference done: {len(pred_phys):,} predictions")

    df = dataset.df.reset_index(drop=True).copy()
    nn_delta = pred_phys[:, 0]
    nn_xptar_raw = pred_phys[:, 1]
    nn_yptar_raw = pred_phys[:, 2]

    # NN → sieve positions using the **full** SHMS optics formula
    # (includes δ corrections for sieve_y, unlike the old linear ×253).
    x_tar = df["P_gtr_x"].to_numpy(dtype=np.float64) if "P_gtr_x" in df.columns else None
    y_tar = df["P_gtr_y"].to_numpy(dtype=np.float64) if "P_gtr_y" in df.columns else None
    nn_sx, nn_sy = project_to_sieve_full(
        x_tar if x_tar is not None else np.zeros_like(nn_xptar_raw),
        y_tar if y_tar is not None else np.zeros_like(nn_yptar_raw),
        nn_xptar_raw,
        nn_yptar_raw,
        nn_delta,
    )
    df["nn_sieve_x"] = nn_sx
    df["nn_sieve_y"] = nn_sy

    # ROOT → sieve positions (exact full formula)
    root_sx, root_sy = project_to_sieve_full(
        df["P_gtr_x"].to_numpy(dtype=np.float64),
        df["P_gtr_y"].to_numpy(dtype=np.float64),
        df["P_gtr_th"].to_numpy(dtype=np.float64),
        df["P_gtr_ph"].to_numpy(dtype=np.float64),
        df["P_gtr_dp"].to_numpy(dtype=np.float64),
    )
    df["root_sieve_x"] = root_sx
    df["root_sieve_y"] = root_sy

    # Mechanical hole design centers (independent ground truth)
    if "candidate_sieve_x_cm" in df.columns:
        df["design_sieve_x"] = df["candidate_sieve_x_cm"].to_numpy(dtype=np.float64)
        df["design_sieve_y"] = df["candidate_sieve_y_cm"].to_numpy(dtype=np.float64)
    else:
        df["design_sieve_x"] = df["weak_hole_xptar_center"].to_numpy(dtype=np.float64) * _SIEVE_DISTANCE_CM
        df["design_sieve_y"] = df["weak_hole_yptar_center"].to_numpy(dtype=np.float64) * _SIEVE_DISTANCE_CM

    print("Generating sieve-x plot...")
    sx = _plot_sieve_comparison(df, output_dir / "sieve_x_comparison_by_foil.png", axis="x")
    sx.to_csv(output_dir / "sieve_x_summary.csv", index=False)

    print("Generating sieve-y plot...")
    sy = _plot_sieve_comparison(df, output_dir / "sieve_y_comparison_by_foil.png", axis="y")
    sy.to_csv(output_dir / "sieve_y_summary.csv", index=False)

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
