#!/usr/bin/env python3
"""
Evaluate a trained model and compare with the polynomial baseline.

Usage
-----
python training/scripts/evaluate_model.py \\
    --checkpoint checkpoints/finetune/best_finetune.pth \\
    --test-data /path/to/test_data.csv \\
    --poly-coeffs models/poly_coeffs_20260112_202706.json \\
    [--scaler-bundle checkpoints/pretrain/scaler_bundle.json] \\
    [--plot-dir plots/] \\
    [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHMS optics — evaluate and compare models")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_finetune.pth (or best_pretrain.pth)",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to test dataset (CSV or Parquet, hcana column names)",
    )
    parser.add_argument(
        "--poly-coeffs",
        default=None,
        help="Path to poly_coeffs_*.json for polynomial baseline comparison",
    )
    parser.add_argument(
        "--scaler-bundle",
        default=None,
        help="Path to scaler_bundle.json; auto-detected from checkpoint dir if omitted",
    )
    parser.add_argument("--p0", type=float, default=None, help="Central momentum GeV/c")
    parser.add_argument("--plot-dir", default=None, help="Directory to save resolution plots")
    parser.add_argument("--device", default="cpu", help="'cuda' or 'cpu'")
    return parser.parse_args()


def main() -> None:
    import torch
    from training.data.sieve_dataset import SieveDataset
    from training.data.preprocessing import ScalerBundle
    from training.models.residual_mlp import ResidualMLP
    from training.evaluation.metrics import OpticsEvaluator

    args = parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})

    model = ResidualMLP(
        input_dim=mcfg.get("input_dim", 6),
        hidden_dim=mcfg.get("hidden_dim", 256),
        n_residual_blocks=mcfg.get("n_residual_blocks", 4),
        branch_dim=mcfg.get("branch_dim", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # Load scaler bundle
    scaler_path = args.scaler_bundle
    if scaler_path is None:
        # Try to find alongside checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint)
        candidate = os.path.join(ckpt_dir, "..", "pretrain", "scaler_bundle.json")
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            scaler_path = candidate

    scaler_bundle = None
    if scaler_path and os.path.exists(scaler_path):
        scaler_bundle = ScalerBundle.load(scaler_path)
        print(f"Loaded scaler bundle from: {scaler_path}")
    else:
        print("Warning: no scaler bundle found — metrics will be in normalised space.")

    # Build dataset
    dataset = SieveDataset(
        data_source=args.test_data,
        p0_value=args.p0,
        scaler_X=scaler_bundle.scaler_X if scaler_bundle else None,
        scaler_Y=scaler_bundle.scaler_Y if scaler_bundle else None,
    )
    print(f"Test dataset size: {len(dataset)} events")

    # Evaluate
    evaluator = OpticsEvaluator(model=model, scaler_bundle=scaler_bundle, device=args.device)
    metrics = evaluator.evaluate(dataset)

    print("\n=== NN Metrics ===")
    for key, val in sorted(metrics.items()):
        print(f"  {key:<20}: {val:.6f}")

    # Polynomial comparison
    if args.poly_coeffs:
        print(f"\n=== NN vs Polynomial ({os.path.basename(args.poly_coeffs)}) ===")
        df_cmp = evaluator.compare_with_polynomial(dataset, args.poly_coeffs)
        print(df_cmp.to_string(index=False))

    # Resolution plots
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        for target in ["delta", "xptar", "yptar", "ytar"]:
            save_path = os.path.join(args.plot_dir, f"resolution_{target}.png")
            evaluator.plot_vertex_resolution(dataset, target=target, save_path=save_path)
            print(f"Saved: {save_path}")
        sieve_path = os.path.join(args.plot_dir, "sieve_reconstruction.png")
        evaluator.plot_sieve_reconstruction(dataset, save_path=sieve_path)
        print(f"Saved: {sieve_path}")


if __name__ == "__main__":
    main()
