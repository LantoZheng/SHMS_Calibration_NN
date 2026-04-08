#!/usr/bin/env python3
"""
Single-file MLP calibration sanity test on SIMC ROOT output.

Default behavior follows the corrected workflow:
- use successful transport events only (stop_id == 0)
- train on core targets [delta, xptar, yptar]
- infer p0 from the .inp file when available
- avoid injecting random x_tar noise by default (x_tar = 0)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import uproot
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataBundle:
    X: np.ndarray
    Y: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    prefix: str
    n_all: int
    n_after_filter: int


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return self.act(x + h)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_blocks: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = self.blocks(h)
        return self.head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file MLP test for SHMS calibration")
    parser.add_argument("--root-file", required=True, help="Path to SIMC ROOT file")
    parser.add_argument("--tree-name", default="h10", help="Tree name in ROOT file")
    parser.add_argument("--inp-file", default=None, help="Optional SIMC .inp file for auto p0 parsing")
    parser.add_argument("--p0", type=float, default=None, help="Override central momentum (GeV/c)")
    parser.add_argument("--x-tar-mode", choices=["zero", "random"], default="zero")
    parser.add_argument("--x-tar-sigma", type=float, default=0.1, help="Gaussian sigma for synthesized x_tar [cm] when x-tar-mode=random")
    parser.add_argument("--target-mode", choices=["core3", "all", "delta-only"], default="core3")
    parser.add_argument("--filter-stop-id", action="store_true", default=True, help="Keep only stop_id == 0 events (default on)")
    parser.add_argument("--no-filter-stop-id", dest="filter_stop_id", action="store_false", help="Disable stop_id filtering")
    parser.add_argument("--max-events", type=int, default=100000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="experiments/mlp_reduced_data_test/outputs",
        help="Where to write metrics/plots",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_prefix(branches: List[str]) -> str:
    candidates = ["ps", "hs"]
    for pref in candidates:
        required = [f"{pref}xfp", f"{pref}yfp", f"{pref}xpfp", f"{pref}ypfp", f"{pref}deltai"]
        if all(r in branches for r in required):
            return pref
    raise RuntimeError("Cannot detect branch prefix. Expected ps* or hs* branches.")


def infer_p0_from_inp(inp_file: Path | None) -> float | None:
    if inp_file is None or not inp_file.exists():
        return None
    target_keywords = ("Spectrometer central momentum", "Spectrometer momentum")
    for line in inp_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if any(k in line for k in target_keywords):
            nums = re.findall(r"[-+]?\d*\.?\d+", line)
            if nums:
                return float(nums[0]) / 1000.0
    return None


def load_data(
    root_file: str,
    tree_name: str,
    target_mode: str,
    p0: float,
    x_tar_mode: str,
    x_tar_sigma: float,
    max_events: int,
    seed: int,
    filter_stop_id: bool,
) -> DataBundle:
    rf: Any = uproot.open(root_file)
    tree = rf[tree_name]
    branches = list(tree.keys())

    pref = detect_prefix(branches)
    input_branches = [f"{pref}xfp", f"{pref}yfp", f"{pref}xpfp", f"{pref}ypfp"]

    if target_mode == "all":
        target_branches = [
            f"{pref}deltai",
            f"{pref}xptari",
            f"{pref}yptari",
            f"{pref}ztari",
        ]
        target_names = ["delta", "xptar", "yptar", "ytar"]
    elif target_mode == "core3":
        target_branches = [
            f"{pref}deltai",
            f"{pref}xptari",
            f"{pref}yptari",
        ]
        target_names = ["delta", "xptar", "yptar"]
    else:
        target_branches = [f"{pref}deltai"]
        target_names = ["delta"]

    wanted = input_branches + target_branches + (["stop_id"] if filter_stop_id else [])
    arr = tree.arrays(wanted, library="np")
    rf.close()

    n_all = len(arr[input_branches[0]])
    n = n_all
    if max_events is not None:
        n = min(n, max_events)

    X_raw = np.column_stack([arr[b][:n].astype(np.float32) for b in input_branches])
    Y_raw = np.column_stack([arr[b][:n].astype(np.float32) for b in target_branches])

    mask = np.isfinite(X_raw).all(axis=1) & np.isfinite(Y_raw).all(axis=1)
    if filter_stop_id:
        mask &= (arr["stop_id"][:n] == 0)

    X_raw = X_raw[mask]
    Y_raw = Y_raw[mask]
    n_after_filter = int(mask.sum())

    if x_tar_mode == "random":
        rng = np.random.default_rng(seed)
        x_tar = rng.normal(0.0, x_tar_sigma, size=(n_after_filter, 1)).astype(np.float32)
    else:
        x_tar = np.zeros((n_after_filter, 1), dtype=np.float32)
    p0_col = np.full((n_after_filter, 1), p0, dtype=np.float32)

    X = np.concatenate([X_raw, x_tar, p0_col], axis=1)

    return DataBundle(
        X=X,
        Y=Y_raw,
        feature_names=["x_fp", "y_fp", "xp_fp", "yp_fp", "x_tar", "p0"],
        target_names=target_names,
        prefix=pref,
        n_all=n_all,
        n_after_filter=n_after_filter,
    )


def fit_and_eval(
    X: np.ndarray,
    Y: np.ndarray,
    val_fraction: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Dict[str, object]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    n_train = len(X_train)

    sx = StandardScaler().fit(X_train)
    sy = StandardScaler().fit(y_train)

    X_train_s = sx.transform(X_train).astype(np.float32)
    X_val_s = sx.transform(X_val).astype(np.float32)
    y_train_s = sy.transform(y_train).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = ResidualMLP(in_dim=X.shape[1], out_dim=Y.shape[1], hidden=256, n_blocks=4, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    x_train_t = torch.from_numpy(X_train_s).to(device)
    y_train_t = torch.from_numpy(y_train_s).to(device)
    x_val_t = torch.from_numpy(X_val_s).to(device)
    y_val_t = torch.from_numpy(sy.transform(y_val).astype(np.float32)).to(device)

    steps = math.ceil(len(x_train_t) / batch_size)
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train_t), device=device)
        epoch_loss = 0.0
        for s in range(steps):
            bidx = perm[s * batch_size : (s + 1) * batch_size]
            xb = x_train_t[bidx]
            yb = y_train_t[bidx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        train_loss_hist.append(epoch_loss / max(steps, 1))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()
        val_loss_hist.append(float(val_loss))

    model.eval()
    with torch.no_grad():
        pred_val_s = model(x_val_t).cpu().numpy()
    pred_val = sy.inverse_transform(pred_val_s)

    err = pred_val - y_val
    rmse = np.sqrt(np.mean(np.square(err), axis=0))
    mae = np.mean(np.abs(err), axis=0)

    return {
        "n_train": int(n_train),
        "n_val": int(len(X_val)),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "y_true": y_val,
        "y_pred": pred_val,
    }


def save_rmse_mae_plot(metric: Dict[str, object], target_names: List[str], out_png: Path) -> None:
    rmse_vals = np.array(metric["rmse"], dtype=float)
    mae_vals = np.array(metric["mae"], dtype=float)
    x = np.arange(len(target_names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, rmse_vals, width=w, label="RMSE")
    ax.bar(x + w / 2, mae_vals, width=w, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.set_title("RMSE / MAE (Filtered stop_id==0, Full Training)")
    ax.set_ylabel("Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_loss_plot(metric: Dict[str, object], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metric["train_loss_hist"], label="train loss")
    ax.plot(metric["val_loss_hist"], label="val loss")
    ax.set_title("Training/Validation Loss Curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inp_file = Path(args.inp_file).expanduser() if args.inp_file else None
    p0_from_inp = infer_p0_from_inp(inp_file)
    p0 = args.p0 if args.p0 is not None else (p0_from_inp if p0_from_inp is not None else 1.4)

    bundle = load_data(
        root_file=args.root_file,
        tree_name=args.tree_name,
        target_mode=args.target_mode,
        p0=p0,
        x_tar_mode=args.x_tar_mode,
        x_tar_sigma=args.x_tar_sigma,
        max_events=args.max_events,
        seed=args.seed,
        filter_stop_id=args.filter_stop_id,
    )

    metric = fit_and_eval(
        X=bundle.X,
        Y=bundle.Y,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print(f"n_train={metric['n_train']} n_val={metric['n_val']} rmse={np.round(metric['rmse'], 6).tolist()}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = args.target_mode

    metrics_path = output_dir / f"metrics_{mode}_{stamp}.json"
    rmse_plot_path = output_dir / f"rmse_mae_{mode}_{stamp}.png"
    loss_plot_path = output_dir / f"loss_{mode}_{stamp}.png"
    summary_path = output_dir / f"run_summary_{mode}_{stamp}.txt"

    payload = {
        "root_file": str(Path(args.root_file).resolve()),
        "inp_file": str(inp_file.resolve()) if inp_file and inp_file.exists() else None,
        "tree_name": args.tree_name,
        "prefix": bundle.prefix,
        "target_mode": args.target_mode,
        "filter_stop_id": args.filter_stop_id,
        "x_tar_mode": args.x_tar_mode,
        "p0_gev": p0,
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
        "max_events": args.max_events,
        "val_fraction": args.val_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "events": {
            "all": bundle.n_all,
            "after_filter": bundle.n_after_filter,
        },
        "metrics": {
            "n_train": metric["n_train"],
            "n_val": metric["n_val"],
            "rmse": metric["rmse"],
            "mae": metric["mae"],
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    save_rmse_mae_plot(metric, bundle.target_names, rmse_plot_path)
    save_loss_plot(metric, loss_plot_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Single-file MLP calibration sanity test\n")
        f.write(f"root_file   : {str(Path(args.root_file).resolve())}\n")
        f.write(f"inp_file    : {str(inp_file.resolve()) if inp_file and inp_file.exists() else 'N/A'}\n")
        f.write(f"prefix      : {bundle.prefix}\n")
        f.write(f"events      : all={bundle.n_all}, after_filter={bundle.n_after_filter}\n")
        f.write(f"filter_stop : {args.filter_stop_id}\n")
        f.write(f"x_tar_mode  : {args.x_tar_mode}\n")
        f.write(f"p0_gev      : {p0}\n")
        f.write(f"target_mode : {args.target_mode}\n")
        f.write(f"targets     : {bundle.target_names}\n")
        f.write(f"n_train     : {metric['n_train']}\n")
        f.write(f"n_val       : {metric['n_val']}\n")
        f.write(f"rmse        : {np.array(metric['rmse']).round(6).tolist()}\n")
        f.write(f"mae         : {np.array(metric['mae']).round(6).tolist()}\n")

    print("\nDone.")
    print(f"Metrics: {metrics_path}")
    print(f"RMSE   : {rmse_plot_path}")
    print(f"Loss   : {loss_plot_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
