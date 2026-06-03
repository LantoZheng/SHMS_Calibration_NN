#!/usr/bin/env python3
"""
Iterative self-consistent labeling-training loop for Stage-2 SHMS transport.

Each iteration:
  1. NN inference on full dataset using current best checkpoint
  2. Replace HCANA sieve_x/sieve_y with NN-predicted sieve positions
  3. Re-run HDBSCAN clustering + mechanical-hole matching
  4. Export new labelled CSV
  5. Train new Stage-2 model with updated labels
  6. Track convergence metrics
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-checkpoint", required=True)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--config-template", required=True)
    p.add_argument("--output-base", required=True)
    p.add_argument("--num-iterations", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--ytar-foil-check-margin-cm", type=float, default=3.0)
    return p.parse_args()


def run_cmd(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*50}\n  {desc}\n{'='*50}")
    print(f"  {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"  {'OK' if ok else 'FAIL'} ({elapsed:.0f}s)")
    return ok


def read_metrics(checkpoint_dir: Path, post_dir: Path, label_summary_path: Path) -> dict:
    m: dict[str, Any] = {}

    best_ckpt = checkpoint_dir / "best_finetune.pth"
    if best_ckpt.exists():
        import torch
        c = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        m["epoch"] = c.get("epoch")
        m["val_loss"] = float(c.get("val_loss", float("nan")))
        m["select_metric"] = float(c.get("selection_metric_value", float("nan")))
        m["select_name"] = str(c.get("selection_metric_name", "?"))

    sieve_csv = post_dir / "sieve_pattern_summary.csv"
    if sieve_csv.exists():
        sd = pd.read_csv(sieve_csv)
        for _, r in sd[sd["method"] == "nn"].iterrows():
            m[f"foil{int(r['foil'])}_sieve_rmse"] = float(r["rmse_radial_cm"])

    ztar_csv = post_dir / "ztar_summary.csv"
    if ztar_csv.exists():
        zd = pd.read_csv(ztar_csv).set_index("slice")
        for sl in ["all", "foil0", "foil1", "foil2"]:
            if sl in zd.index:
                m[f"{sl}_nn_rmse_cm"] = float(zd.loc[sl, "nn_rmse_cm"])

    if label_summary_path.exists():
        ls = json.load(open(label_summary_path))
        m["n_labeled"] = ls.get("n_labeled_events")
        m["n_holes"] = ls.get("hole_count_total")
        cm = ls.get("cluster_to_mechanical_match", {})
        m["duplicate_holes"] = cm.get("duplicate_mechanical_holes", -1)

    return m


def main():
    args = parse_args()
    base = Path(args.output_base); base.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.start_checkpoint)
    csv_path = Path(args.input_csv)
    cfg_tpl = Path(args.config_template)

    with open(cfg_tpl) as fh:
        cfg = yaml.safe_load(fh)

    scaler_bundle_path = str((_REPO_ROOT / cfg["pretrained"]["scaler_bundle_path"]).resolve())

    all_metrics: list[dict] = []

    for it in range(1, args.num_iterations + 1):
        print(f"\n{'#'*50}\n  ITERATION {it}/{args.num_iterations}\n{'#'*50}")

        # Step 1: Relabel
        relabel_dir = base / "dataset" / f"iter{it}"
        relabel_dir.mkdir(parents=True, exist_ok=True)
        relabel_csv = relabel_dir / "stage2_nnrelabel.csv"
        relabel_sum = relabel_dir / "stage2_nnrelabel_summary.json"

        relabel_script = str(_REPO_ROOT / "training" / "scripts" / "relabel_stage2_with_nn_sieve.py")
        relabel_cmd = [
            sys.executable, relabel_script,
            "--checkpoint", str(ckpt),
            "--input-csv", str(csv_path),
            "--output-csv", str(relabel_csv),
            "--summary-json", str(relabel_sum),
            "--scaler-bundle", scaler_bundle_path,
            "--device", args.device,
        ]
        if args.max_events:
            relabel_cmd += ["--max-events", str(args.max_events)]
        if float(args.ytar_foil_check_margin_cm) > 0:
            relabel_cmd += ["--ytar-foil-check-margin-cm", str(args.ytar_foil_check_margin_cm)]

        ok = run_cmd(relabel_cmd, f"Relabel iteration {it}")
        if not ok: break

        if relabel_sum.exists():
            rs = json.load(open(relabel_sum))
            cm = rs.get("cluster_to_mechanical_match", {})
            print(f"  Labeled={rs.get('n_labeled_events')}, holes={rs.get('hole_count_total')}, "
                  f"dups={cm.get('duplicate_mechanical_holes')}")

        # Step 2: Train
        iter_cfg_path = base / f"config_iter{it}.yaml"
        ckpt_dir = base / "checkpoints" / f"iter{it}"
        post_dir = base / "postprocess" / f"iter{it}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        post_dir.mkdir(parents=True, exist_ok=True)

        cfg["output"]["checkpoint_dir"] = str(ckpt_dir)
        cfg["postprocess"]["output_dir"] = str(post_dir)
        with open(iter_cfg_path, "w") as fh:
            yaml.dump(cfg, fh)

        train_script = str(_REPO_ROOT / "training" / "scripts" / "run_stage2_transport_fullroot.py")
        train_cmd = [
            sys.executable, train_script,
            "--config", str(iter_cfg_path),
            "--root-file", str(relabel_csv),
            "--device", args.device,
        ]
        if args.max_events:
            train_cmd += ["--max-events", str(args.max_events)]

        ok = run_cmd(train_cmd, f"Train iteration {it}")
        if not ok: break

        # Step 3: Collect metrics
        metrics = read_metrics(ckpt_dir, post_dir, relabel_sum)
        metrics["iteration"] = it
        all_metrics.append(metrics)

        sel = metrics.get("select_metric", float("nan"))
        f0 = metrics.get("foil0_sieve_rmse", float("nan"))
        f1 = metrics.get("foil1_sieve_rmse", float("nan"))
        f2 = metrics.get("foil2_sieve_rmse", float("nan"))
        print(f"  select={sel:.6f}, sieve=[{f0:.4f}, {f1:.4f}, {f2:.4f}]")

        # Update for next iteration
        best = ckpt_dir / "best_finetune.pth"
        if best.exists():
            ckpt = best
        csv_path = relabel_csv

    # Convergence summary
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(base / "convergence.csv", index=False)
        with open(base / "convergence.json", "w") as fh:
            json.dump(all_metrics, fh, ensure_ascii=False, indent=2)

        print(f"\n{'='*50}\n  CONVERGENCE SUMMARY\n{'='*50}")
        key_cols = [c for c in [
            "iteration", "select_metric", "foil0_sieve_rmse", "foil1_sieve_rmse",
            "foil2_sieve_rmse", "all_nn_rmse_cm", "n_labeled", "n_holes", "duplicate_holes"
        ] if c in df.columns]
        print(df[key_cols].to_string(index=False))
        print(f"\n  Full: {base / 'convergence.csv'}")


if __name__ == "__main__":
    main()
