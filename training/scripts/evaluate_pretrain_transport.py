#!/usr/bin/env python3
"""Evaluate a transport-style pretrain checkpoint on SIMC ROOT data.

This script mirrors the diagnostics used in
`experiments/ResMLP/ResMLP_transport.ipynb`, but runs on the formal training
artifacts (`best_pretrain.pth`, `scaler_bundle.json`, `training_history_pretrain.json`).

Outputs
-------
- metrics_summary.csv
- metrics_payload.json
- nonlinear_diagnostics.csv
- relative_rmse_comparison.png
- loss_curve.png
- parity.png
- residual.png
- nonlinear_correction_parity.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from training.data.preprocessing import ScalerBundle
from training.data.simc_dataset import _resolve_branch_group, _resolve_fry_branch
from training.models import build_model_from_config

plt.style.use("seaborn-v0_8-whitegrid")

_TARGET_NAMES = ["delta", "xptar", "yptar", "ytar"]


@dataclass
class EvalBundle:
    X_raw: np.ndarray
    Y_raw: np.ndarray
    Y_reco: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    target_truth_branches: list[str]
    target_reco_branches: list[str | None]
    prefix: str
    total_events: int
    events_after_filter: int
    filter_stop_id: bool


@dataclass
class EvalResults:
    val_indices: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_linear: np.ndarray
    y_pred_correction: np.ndarray
    y_true_nonlinear: np.ndarray
    x_val_raw: np.ndarray
    root_metrics: dict[str, Any] | None
    nn_metrics: dict[str, Any]
    linear_metrics: dict[str, Any]
    correction_metrics: dict[str, Any]
    rmse_improvement_vs_linear_pct: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SHMS pretrain checkpoint with notebook-style diagnostics")
    parser.add_argument("--checkpoint", required=True, help="Path to best_pretrain.pth")
    parser.add_argument("--simc-files", nargs="+", required=True, help="ROOT files or glob patterns")
    parser.add_argument("--scaler-bundle", default=None, help="Path to scaler_bundle.json (auto-detect if omitted)")
    parser.add_argument("--tree-name", default=None, help="ROOT tree name (default from checkpoint config or h10)")
    parser.add_argument("--output-dir", default=None, help="Directory for plots/metrics (default: checkpoint_dir/eval_transport_<timestamp>)")
    parser.add_argument("--device", default=None, help="cuda/cpu; defaults to cuda if available")
    parser.add_argument("--batch-size", type=int, default=8192, help="Evaluation batch size")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on events before split")
    parser.add_argument("--random-seed", type=int, default=None, help="Override checkpoint training.random_seed")
    parser.add_argument("--val-fraction", type=float, default=None, help="Override checkpoint training.val_fraction")
    parser.add_argument("--fry-branch", default=None, help="Override fry branch name")
    parser.add_argument("--x-tar-mode", choices=["zero", "random"], default="zero", help="Used only if feature schema requests x_tar")
    parser.add_argument("--x-tar-sigma", type=float, default=0.1, help="Sigma for synthetic x_tar if requested")
    parser.add_argument("--p0", type=float, default=None, help="Constant p0 value if feature schema requests p0")
    parser.add_argument("--filter-stop-id", action="store_true", help="Apply stop_id == 0 filter before evaluation")
    return parser.parse_args()


def detect_prefix(branches: list[str]) -> str:
    for pref in ["ps", "hs"]:
        required = [f"{pref}xfp", f"{pref}yfp", f"{pref}xpfp", f"{pref}ypfp", f"{pref}deltai"]
        if all(name in branches for name in required):
            return pref
    raise RuntimeError("Cannot detect branch prefix. Expected ps* or hs* branches.")


def infer_reco_branch(pref: str, truth_branch: str, all_branches: list[str]) -> str | None:
    stripped = truth_branch[len(pref):] if truth_branch.startswith(pref) else truth_branch
    manual = {
        "deltai": [f"{pref}delta", f"{pref}deltar"],
        "xptari": [f"{pref}xptar"],
        "yptari": [f"{pref}yptar"],
        "ztari": [f"{pref}ztar", f"{pref}ytar"],
    }
    candidates: list[str] = list(manual.get(stripped, []))
    if stripped.endswith("i"):
        candidates.append(f"{pref}{stripped[:-1]}")
    for candidate in candidates:
        if candidate in all_branches:
            return candidate
    return None


def align_reco_to_truth(pref: str, truth_branch: str, reco_branch: str | None, reco_values: np.ndarray) -> tuple[np.ndarray, str | None]:
    if reco_branch is None:
        return reco_values, None
    if truth_branch == f"{pref}ztari" and reco_branch in {f"{pref}ztar", f"{pref}ytar"}:
        return -reco_values, f"-{reco_branch}"
    return reco_values, reco_branch


def expand_root_paths(patterns: list[str]) -> list[str]:
    expanded: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded.extend(matches)
        elif os.path.exists(pattern):
            expanded.append(pattern)
    expanded = sorted(dict.fromkeys(os.path.abspath(path) for path in expanded))
    if not expanded:
        raise FileNotFoundError("No ROOT files matched the provided --simc-files inputs.")
    return expanded


def load_eval_bundle(
    root_paths: list[str],
    tree_name: str,
    feature_schema: list[str],
    fry_branch: str | None,
    max_events: int | None,
    rng_seed: int,
    x_tar_mode: str,
    x_tar_sigma: float,
    p0: float | None,
    filter_stop_id: bool,
) -> EvalBundle:
    import uproot

    target_truth_keys = list(_TARGET_NAMES)
    target_names = list(_TARGET_NAMES)

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    y_reco_chunks: list[np.ndarray] = []
    total_events = 0
    total_after_filter = 0
    prefix_seen: str | None = None
    truth_branch_labels: list[str] | None = None
    reco_branch_labels: list[str | None] | None = None
    resolved_feature_names: list[str] | None = None
    rng = np.random.default_rng(rng_seed)

    remaining = max_events

    for root_path in root_paths:
        with uproot.open(root_path) as root_file:
            tree = root_file[tree_name]
            branches = list(tree.keys())
            pref = detect_prefix(branches)
            prefix_seen = prefix_seen or pref
            if prefix_seen != pref:
                raise RuntimeError(f"Mixed branch prefixes are not supported: got {prefix_seen} and {pref}")

            resolved_inputs = _resolve_branch_group(["x_fp", "y_fp", "xp_fp", "yp_fp"], branches)
            resolved_targets = _resolve_branch_group(target_truth_keys, branches)
            resolved_fry = _resolve_fry_branch(branches, fry_branch) if "fry" in feature_schema else None
            raw_target_reco = [infer_reco_branch(pref, resolved_targets[name], branches) for name in target_truth_keys]
            reco_existing = [name for name in raw_target_reco if name is not None]

            wanted = list(dict.fromkeys([
                *resolved_inputs.values(),
                *resolved_targets.values(),
                *reco_existing,
                *([resolved_fry] if resolved_fry else []),
                *( ["stop_id"] if filter_stop_id and "stop_id" in branches else []),
            ]))
            arr = tree.arrays(wanted, library="np")

        n_file = len(arr[next(iter(resolved_inputs.values()))])
        take = n_file if remaining is None else min(n_file, remaining)
        if take <= 0:
            break

        x_raw = np.column_stack([arr[resolved_inputs[key]][:take].astype(np.float32) for key in ["x_fp", "y_fp", "xp_fp", "yp_fp"]])
        feature_names = ["x_fp", "y_fp", "xp_fp", "yp_fp"]
        if resolved_fry is not None:
            x_raw = np.concatenate([x_raw, arr[resolved_fry][:take].astype(np.float32).reshape(-1, 1)], axis=1)
            feature_names.append("fry")
        if "x_tar" in feature_schema:
            if x_tar_mode == "random":
                x_tar = rng.normal(0.0, x_tar_sigma, size=(take, 1)).astype(np.float32)
            else:
                x_tar = np.zeros((take, 1), dtype=np.float32)
            x_raw = np.concatenate([x_raw, x_tar], axis=1)
            feature_names.append("x_tar")
        if "p0" in feature_schema:
            if p0 is None:
                raise ValueError("Feature schema requests p0, but no --p0 value was provided.")
            p0_col = np.full((take, 1), p0, dtype=np.float32)
            x_raw = np.concatenate([x_raw, p0_col], axis=1)
            feature_names.append("p0")

        if feature_names != feature_schema:
            raise RuntimeError(f"Resolved feature order {feature_names} does not match feature schema {feature_schema}.")

        y_raw = np.column_stack([arr[resolved_targets[key]][:take].astype(np.float32) for key in target_truth_keys])

        reco_cols: list[np.ndarray] = []
        reco_labels_now: list[str | None] = []
        for truth_key, reco_branch in zip(target_truth_keys, raw_target_reco):
            if reco_branch is not None:
                reco_values, reco_label = align_reco_to_truth(pref, resolved_targets[truth_key], reco_branch, arr[reco_branch][:take].astype(np.float32))
                reco_cols.append(reco_values)
                reco_labels_now.append(reco_label)
            else:
                reco_cols.append(np.full(take, np.nan, dtype=np.float32))
                reco_labels_now.append(None)
        y_reco = np.column_stack(reco_cols)

        mask = np.isfinite(x_raw).all(axis=1) & np.isfinite(y_raw).all(axis=1)
        if filter_stop_id and "stop_id" in arr:
            mask &= (arr["stop_id"][:take] == 0)

        x_raw = x_raw[mask]
        y_raw = y_raw[mask]
        y_reco = y_reco[mask]

        x_chunks.append(x_raw)
        y_chunks.append(y_raw)
        y_reco_chunks.append(y_reco)
        total_events += int(take)
        total_after_filter += int(mask.sum())
        truth_branch_labels = [resolved_targets[key] for key in target_truth_keys]
        reco_branch_labels = reco_labels_now
        resolved_feature_names = feature_names

        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break

    if not x_chunks:
        raise RuntimeError("No events were loaded for evaluation.")

    return EvalBundle(
        X_raw=np.concatenate(x_chunks, axis=0),
        Y_raw=np.concatenate(y_chunks, axis=0),
        Y_reco=np.concatenate(y_reco_chunks, axis=0),
        feature_names=list(resolved_feature_names or feature_schema),
        target_names=target_names,
        target_truth_branches=list(truth_branch_labels or target_truth_keys),
        target_reco_branches=list(reco_branch_labels or [None] * len(target_names)),
        prefix=str(prefix_seen),
        total_events=total_events,
        events_after_filter=total_after_filter,
        filter_stop_id=filter_stop_id,
    )


def compute_error_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    err = pred - truth
    n_targets = truth.shape[1]
    rmse = np.full(n_targets, np.nan, dtype=np.float64)
    mae = np.full(n_targets, np.nan, dtype=np.float64)
    bias = np.full(n_targets, np.nan, dtype=np.float64)
    resid_std = np.full(n_targets, np.nan, dtype=np.float64)
    rel_rmse = np.full(n_targets, np.nan, dtype=np.float64)
    rel_mae = np.full(n_targets, np.nan, dtype=np.float64)

    for i in range(n_targets):
        mask = np.isfinite(pred[:, i]) & np.isfinite(truth[:, i])
        if not np.any(mask):
            continue
        e = err[mask, i]
        y = truth[mask, i]
        rmse[i] = np.sqrt(np.mean(np.square(e)))
        mae[i] = np.mean(np.abs(e))
        bias[i] = np.mean(e)
        resid_std[i] = np.std(e)
        scale = max(np.mean(np.abs(y)), 1e-8)
        rel_rmse[i] = rmse[i] / scale
        rel_mae[i] = mae[i] / scale

    return {
        "err": err,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "resid_std": resid_std,
        "rel_rmse": rel_rmse,
        "rel_mae": rel_mae,
    }


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_val_indices(n_events: int, val_fraction: float, seed: int) -> np.ndarray:
    n_val = max(1, int(n_events * val_fraction))
    n_train = n_events - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_events, generator=generator).cpu().numpy()
    return perm[n_train:]


@torch.no_grad()
def run_model(model: torch.nn.Module, x_scaled: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    total_chunks: list[np.ndarray] = []
    linear_chunks: list[np.ndarray] = []
    corr_chunks: list[np.ndarray] = []

    model.eval()
    for start in range(0, len(xs), batch_size):
        xb = xs[start : start + batch_size]
        outputs = model(xb)
        total = torch.cat([outputs[name] for name in _TARGET_NAMES], dim=1)
        linear = outputs.get("linear_output")
        correction = outputs.get("correction")
        if linear is None:
            linear = total
        if correction is None:
            correction = torch.zeros_like(total)

        total_chunks.append(total.cpu().numpy())
        linear_chunks.append(linear.cpu().numpy())
        corr_chunks.append(correction.cpu().numpy())

    return (
        np.concatenate(total_chunks, axis=0),
        np.concatenate(linear_chunks, axis=0),
        np.concatenate(corr_chunks, axis=0),
    )


def invert_transport_outputs(scaler_bundle: ScalerBundle, total_s: np.ndarray, linear_s: np.ndarray, corr_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.asarray(scaler_bundle.scaler_Y.scale_, dtype=np.float64).reshape(1, -1)
    mean = np.asarray(scaler_bundle.scaler_Y.mean_, dtype=np.float64).reshape(1, -1)
    total = mean + total_s * scale
    linear = mean + linear_s * scale
    correction = corr_s * scale
    return total, linear, correction


def make_summary_row(bundle: EvalBundle, results: EvalResults, best_epoch: int, best_val_loss: float) -> pd.DataFrame:
    row: dict[str, Any] = {
        "n_events_total": bundle.total_events,
        "n_events_after_filter": bundle.events_after_filter,
        "n_val": len(results.val_indices),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    for i, target in enumerate(bundle.target_names):
        row[f"linear_rmse_{target}"] = results.linear_metrics["rmse"][i]
        row[f"model_rmse_{target}"] = results.nn_metrics["rmse"][i]
        row[f"root_rmse_{target}"] = results.root_metrics["rmse"][i] if results.root_metrics is not None else np.nan
        row[f"linear_mae_{target}"] = results.linear_metrics["mae"][i]
        row[f"model_mae_{target}"] = results.nn_metrics["mae"][i]
        row[f"root_mae_{target}"] = results.root_metrics["mae"][i] if results.root_metrics is not None else np.nan
        row[f"rel_rmse_model_{target}_pct"] = 100.0 * results.nn_metrics["rel_rmse"][i]
        row[f"rmse_improve_vs_linear_{target}_pct"] = results.rmse_improvement_vs_linear_pct[i]
        row[f"correction_rmse_{target}"] = results.correction_metrics["rmse"][i]

    return pd.DataFrame([row])


def make_nonlinear_diag(bundle: EvalBundle, results: EvalResults) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for t_idx, target_name in enumerate(bundle.target_names):
        nonlinear_truth = results.y_true_nonlinear[:, t_idx]
        nonlinear_pred = results.y_pred_correction[:, t_idx]
        corr_truth_vs_pred = safe_corrcoef(nonlinear_truth, nonlinear_pred)
        for f_idx, feature_name in enumerate(bundle.feature_names):
            feature_values = results.x_val_raw[:, f_idx]
            rows.append(
                {
                    "target": target_name,
                    "source": feature_name,
                    "feature_corr_with_true_nonlinear": safe_corrcoef(feature_values, nonlinear_truth),
                    "feature_corr_with_predicted_correction": safe_corrcoef(feature_values, nonlinear_pred),
                    "true_vs_predicted_correction_corr": corr_truth_vs_pred,
                }
            )
    return pd.DataFrame(rows)


def plot_relative_rmse(bundle: EvalBundle, results: EvalResults, save_path: Path) -> None:
    linear_rel_rmse_pct = [100.0 * results.linear_metrics["rel_rmse"][i] for i in range(len(bundle.target_names))]
    model_rel_rmse_pct = [100.0 * results.nn_metrics["rel_rmse"][i] for i in range(len(bundle.target_names))]
    root_rel_rmse_pct = [100.0 * results.root_metrics["rel_rmse"][i] for i in range(len(bundle.target_names))] if results.root_metrics is not None else None

    x = np.arange(len(bundle.target_names))
    w = 0.25 if root_rel_rmse_pct is not None else 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    if root_rel_rmse_pct is not None:
        ax.bar(x - w, linear_rel_rmse_pct, width=w, label="Internal linear path", color="gray")
        ax.bar(x, root_rel_rmse_pct, width=w, label="ROOT reco", color="black", alpha=0.65)
        ax.bar(x + w, model_rel_rmse_pct, width=w, label="Transport ResMLP", color="tab:blue")
    else:
        ax.bar(x - w / 2, linear_rel_rmse_pct, width=w, label="Internal linear path", color="gray")
        ax.bar(x + w / 2, model_rel_rmse_pct, width=w, label="Transport ResMLP", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(bundle.target_names)
    ax.set_title("Relative RMSE comparison")
    ax.set_ylabel("Relative Error (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history: dict[str, Any] | None, best_epoch: int, linear_warmup_epochs: int, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if history is not None:
        ax.plot(history.get("train_loss", []), label="train loss")
        ax.plot(history.get("val_loss", []), label="val loss")
    if best_epoch > 0:
        ax.axvline(best_epoch - 1, color="k", linestyle="--", linewidth=1, label=f"best epoch = {best_epoch}")
    if linear_warmup_epochs > 0:
        ax.axvline(linear_warmup_epochs - 1, color="tab:orange", linestyle=":", linewidth=1, label="linear warmup end")
    ax.set_title("Training/Validation Loss Curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_parity(bundle: EvalBundle, results: EvalResults, save_path: Path) -> None:
    nt = results.y_true.shape[1]
    fig, axes = plt.subplots(1, nt, figsize=(5 * nt, 4))
    axes_list = list(axes.ravel()) if isinstance(axes, np.ndarray) else [axes]
    y_root = bundle.Y_reco[results.val_indices]

    for i, target_name in enumerate(bundle.target_names):
        ax = axes_list[i]
        yt = results.y_true[:, i]
        yl = results.y_linear[:, i]
        yp = results.y_pred[:, i]
        ax.scatter(yt, yl, s=4, alpha=0.18, label="linear path")
        ax.scatter(yt, yp, s=4, alpha=0.18, label="corrected output")
        yr = y_root[:, i]
        if np.any(np.isfinite(yr)):
            ax.scatter(yt, yr, s=4, alpha=0.12, label="ROOT reco")
        lo = min(yt.min(), yl.min(), yp.min())
        hi = max(yt.max(), yl.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"Parity: {target_name}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residual(bundle: EvalBundle, results: EvalResults, save_path: Path) -> None:
    res_model = results.y_pred - results.y_true
    res_linear = results.y_linear - results.y_true
    res_root = bundle.Y_reco[results.val_indices] - results.y_true
    nt = res_model.shape[1]
    fig, axes = plt.subplots(1, nt, figsize=(5 * nt, 4))
    axes_list = list(axes.ravel()) if isinstance(axes, np.ndarray) else [axes]

    for i, target_name in enumerate(bundle.target_names):
        ax = axes_list[i]
        ax.hist(res_model[:, i], bins=80, alpha=0.55, label="Transport ResMLP residual")
        ax.hist(res_linear[:, i], bins=80, histtype="step", color="gray", linewidth=1.8, label="Linear path residual")
        rr = res_root[:, i]
        if np.any(np.isfinite(rr)):
            ax.hist(rr[np.isfinite(rr)], bins=80, histtype="step", color="k", linestyle="--", linewidth=1.4, label="ROOT reco residual")
        ax.axvline(0.0, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"Residual: {target_name}")
        ax.set_xlabel("pred - true")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_nonlinear_correction(bundle: EvalBundle, results: EvalResults, save_path: Path) -> None:
    fig, axes = plt.subplots(1, len(bundle.target_names), figsize=(5 * len(bundle.target_names), 4))
    axes_list = list(axes.ravel()) if isinstance(axes, np.ndarray) else [axes]
    for i, target_name in enumerate(bundle.target_names):
        ax = axes_list[i]
        yt_corr = results.y_true_nonlinear[:, i]
        yp_corr = results.y_pred_correction[:, i]
        ax.scatter(yt_corr, yp_corr, s=5, alpha=0.25)
        lo = min(yt_corr.min(), yp_corr.min())
        hi = max(yt_corr.max(), yp_corr.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"Nonlinear correction: {target_name}")
        ax.set_xlabel("true nonlinear part")
        ax.set_ylabel("predicted correction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))
    training_cfg = dict(config.get("training", {}))
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))

    scaler_path = Path(args.scaler_bundle).resolve() if args.scaler_bundle else checkpoint_path.parent / "scaler_bundle.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler bundle not found: {scaler_path}")
    scaler_bundle = ScalerBundle.load(str(scaler_path))

    feature_schema = list(data_cfg.get("feature_schema") or scaler_bundle.input_features)
    tree_name = args.tree_name or data_cfg.get("simc_tree_name", "h10")
    val_fraction = float(args.val_fraction if args.val_fraction is not None else training_cfg.get("val_fraction", 0.2))
    random_seed = int(args.random_seed if args.random_seed is not None else training_cfg.get("random_seed", 42))
    fry_branch = args.fry_branch if args.fry_branch is not None else data_cfg.get("fry_branch")
    linear_warmup_epochs = int(model_cfg.get("linear_warmup_epochs", 0))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint_path.parent / f"eval_transport_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    root_paths = expand_root_paths(args.simc_files)
    bundle = load_eval_bundle(
        root_paths=root_paths,
        tree_name=tree_name,
        feature_schema=feature_schema,
        fry_branch=fry_branch,
        max_events=args.max_events,
        rng_seed=random_seed,
        x_tar_mode=args.x_tar_mode,
        x_tar_sigma=float(args.x_tar_sigma),
        p0=args.p0,
        filter_stop_id=bool(args.filter_stop_id),
    )

    x_scaled = scaler_bundle.transform_X(bundle.X_raw).astype(np.float32)
    val_indices = build_val_indices(len(x_scaled), val_fraction, random_seed)
    x_val_scaled = x_scaled[val_indices]
    y_val = bundle.Y_raw[val_indices]

    model = build_model_from_config(model_cfg, input_dim=len(feature_schema))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    total_s, linear_s, corr_s = run_model(model, x_val_scaled, device=device, batch_size=args.batch_size)
    y_pred, y_linear, y_pred_correction = invert_transport_outputs(scaler_bundle, total_s, linear_s, corr_s)
    y_true_nonlinear = y_val - y_linear

    nn_metrics = compute_error_metrics(y_pred, y_val)
    linear_metrics = compute_error_metrics(y_linear, y_val)
    correction_metrics = compute_error_metrics(y_pred_correction, y_true_nonlinear)
    root_metrics = None
    if np.any(np.isfinite(bundle.Y_reco[val_indices])):
        root_metrics = compute_error_metrics(bundle.Y_reco[val_indices], y_val)

    rmse_improvement_vs_linear_pct = 100.0 * (linear_metrics["rmse"] - nn_metrics["rmse"]) / np.maximum(linear_metrics["rmse"], 1e-8)
    results = EvalResults(
        val_indices=val_indices,
        y_true=y_val,
        y_pred=y_pred,
        y_linear=y_linear,
        y_pred_correction=y_pred_correction,
        y_true_nonlinear=y_true_nonlinear,
        x_val_raw=bundle.X_raw[val_indices],
        root_metrics=root_metrics,
        nn_metrics=nn_metrics,
        linear_metrics=linear_metrics,
        correction_metrics=correction_metrics,
        rmse_improvement_vs_linear_pct=rmse_improvement_vs_linear_pct,
    )

    nonlinear_diag = make_nonlinear_diag(bundle, results)
    summary_df = make_summary_row(bundle, results, best_epoch=int(ckpt.get("epoch", -1)), best_val_loss=float(ckpt.get("val_loss", float("nan"))))

    history_path = checkpoint_path.parent / "training_history_pretrain.json"
    history = None
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as fh:
            history = json.load(fh)

    payload = {
        "checkpoint": str(checkpoint_path),
        "root_files": root_paths,
        "scaler_bundle": str(scaler_path),
        "tree_name": tree_name,
        "device": str(device),
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "target_truth_branches": bundle.target_truth_branches,
        "target_reco_branches": bundle.target_reco_branches,
        "filter_stop_id": bundle.filter_stop_id,
        "n_events_total": bundle.total_events,
        "n_events_after_filter": bundle.events_after_filter,
        "n_val": len(val_indices),
        "val_fraction": val_fraction,
        "random_seed": random_seed,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_val_loss": float(ckpt.get("val_loss", float("nan"))),
        "metrics": {
            "nn_rmse": results.nn_metrics["rmse"].tolist(),
            "nn_mae": results.nn_metrics["mae"].tolist(),
            "nn_bias": results.nn_metrics["bias"].tolist(),
            "nn_resid_std": results.nn_metrics["resid_std"].tolist(),
            "nn_rel_rmse": results.nn_metrics["rel_rmse"].tolist(),
            "linear_rmse": results.linear_metrics["rmse"].tolist(),
            "linear_mae": results.linear_metrics["mae"].tolist(),
            "correction_rmse": results.correction_metrics["rmse"].tolist(),
            "rmse_improvement_vs_linear_pct": results.rmse_improvement_vs_linear_pct.tolist(),
            "root_rmse": results.root_metrics["rmse"].tolist() if results.root_metrics is not None else None,
            "root_mae": results.root_metrics["mae"].tolist() if results.root_metrics is not None else None,
        },
        "nonlinear_diag_preview": nonlinear_diag.head(20).to_dict(orient="records"),
    }

    metrics_json = output_dir / "metrics_payload.json"
    metrics_csv = output_dir / "metrics_summary.csv"
    nonlinear_csv = output_dir / "nonlinear_diagnostics.csv"
    plot_rmse_path = output_dir / "relative_rmse_comparison.png"
    plot_loss_path = output_dir / "loss_curve.png"
    plot_parity_path = output_dir / "parity.png"
    plot_residual_path = output_dir / "residual.png"
    plot_nonlinear_path = output_dir / "nonlinear_correction_parity.png"

    summary_df.to_csv(metrics_csv, index=False)
    nonlinear_diag.to_csv(nonlinear_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    plot_relative_rmse(bundle, results, plot_rmse_path)
    plot_loss_curve(history, best_epoch=int(ckpt.get("epoch", -1)), linear_warmup_epochs=linear_warmup_epochs, save_path=plot_loss_path)
    plot_parity(bundle, results, plot_parity_path)
    plot_residual(bundle, results, plot_residual_path)
    plot_nonlinear_correction(bundle, results, plot_nonlinear_path)

    print(f"Evaluation output directory: {output_dir}")
    print(f"Events: total={bundle.total_events}, after_filter={bundle.events_after_filter}, val={len(val_indices)}")
    print(f"Checkpoint best epoch: {ckpt.get('epoch')}  val_loss={float(ckpt.get('val_loss', float('nan'))):.6f}")
    print("NN RMSE:", np.round(results.nn_metrics["rmse"], 6).tolist())
    print("Linear RMSE:", np.round(results.linear_metrics["rmse"], 6).tolist())
    if results.root_metrics is not None:
        print("ROOT reco RMSE:", np.round(results.root_metrics["rmse"], 6).tolist())
    print("RMSE improvement vs linear (%):", np.round(results.rmse_improvement_vs_linear_pct, 4).tolist())
    print("Saved:")
    for path in [metrics_json, metrics_csv, nonlinear_csv, plot_rmse_path, plot_loss_path, plot_parity_path, plot_residual_path, plot_nonlinear_path]:
        print("-", path)


if __name__ == "__main__":
    main()
