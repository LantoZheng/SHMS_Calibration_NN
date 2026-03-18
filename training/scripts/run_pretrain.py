#!/usr/bin/env python3
"""
Stage 1 — SIMC pre-training entry point.

Usage
-----
python training/scripts/run_pretrain.py \\
    --config training/configs/pretrain_config.yaml \\
    --simc-files /path/to/simc_run_*.root \\
    --output-dir checkpoints/pretrain/ \\
    [--p0 4.4] \\
    [--device cuda]

All ROOT files must be pre-expanded (no shell glob expansion here).
Pass multiple files by repeating --simc-files or using shell expansion:
    --simc-files run1.root run2.root run3.root
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
import os

import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHMS optics — Stage 1 SIMC pre-training")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pretrain_config.yaml",
    )
    parser.add_argument(
        "--simc-files",
        nargs="+",
        required=True,
        help="SIMC ROOT file paths (glob patterns supported)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override checkpoint output directory from config",
    )
    parser.add_argument("--p0", type=float, default=None, help="Central momentum GeV/c")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'")
    parser.add_argument(
        "--max-events", type=int, default=None, help="Cap on number of events per file"
    )
    return parser.parse_args()


def main() -> None:
    import yaml
    from training.data.simc_dataset import SIMCDataset
    from training.data.preprocessing import ScalerBundle
    from training.models.residual_mlp import ResidualMLP
    from training.models.physics_loss import PhysicsInformedLoss
    from training.trainers.pretrain import PretrainTrainer

    args = parse_args()

    # Load config
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # Expand any glob patterns in simc_files
    root_files: list = []
    for pattern in args.simc_files:
        expanded = _glob.glob(pattern)
        if not expanded:
            sys.exit(f"No files matched: {pattern}")
        root_files.extend(sorted(expanded))

    print(f"Found {len(root_files)} SIMC ROOT file(s).")

    # Override config with CLI arguments
    if args.output_dir:
        config.setdefault("output", {})["checkpoint_dir"] = args.output_dir
    checkpoint_dir = config.get("output", {}).get("checkpoint_dir", "checkpoints/pretrain/")
    scaler_path = config.get("output", {}).get("scaler_save_path", os.path.join(checkpoint_dir, "scaler_bundle.json"))

    p0 = args.p0 if args.p0 is not None else None
    tree_name = config.get("data", {}).get("simc_tree_name", "h10")
    x_tar_sigma = config.get("data", {}).get("x_tar_sigma_cm", 0.1)

    # Build dataset — fit scalers on this data
    print("Loading SIMC data …")
    dataset = SIMCDataset(
        root_file_paths=root_files,
        tree_name=tree_name,
        p0_value=p0,
        max_events=args.max_events,
        fit_scalers=True,
        x_tar_sigma_cm=x_tar_sigma,
        rng_seed=config.get("training", {}).get("random_seed", 42),
    )
    print(f"Dataset size: {len(dataset)} events")

    # Save scaler bundle
    mcfg = config.get("model", {})
    input_features = ["x_fp", "y_fp", "xp_fp", "yp_fp", "x_tar"] + (["p0"] if p0 is not None else [])
    target_features = ["delta", "xptar", "yptar", "ytar"]
    bundle = ScalerBundle(input_features=input_features, target_features=target_features)
    if dataset.scaler_X is not None and dataset.scaler_Y is not None:
        bundle.set_fitted_scalers(dataset.scaler_X, dataset.scaler_Y)
        bundle.save(scaler_path)
        print(f"Scaler bundle saved to: {scaler_path}")

    # Build model
    model = ResidualMLP(
        input_dim=mcfg.get("input_dim", 6),
        hidden_dim=mcfg.get("hidden_dim", 256),
        n_residual_blocks=mcfg.get("n_residual_blocks", 4),
        branch_dim=mcfg.get("branch_dim", 64),
    )
    model.model_summary()

    # Build loss
    lcfg = config.get("loss", {})
    transport_matrix = lcfg.get("transport_matrix", None)
    loss_fn = PhysicsInformedLoss(
        lambda_physics=lcfg.get("lambda_physics", 0.01),
        use_huber=lcfg.get("use_huber", True),
        huber_delta=lcfg.get("huber_delta", 1.0),
        transport_matrix=transport_matrix if transport_matrix else None,
    )

    # Train
    trainer = PretrainTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=args.device,
    )
    history = trainer.train(train_dataset=dataset, checkpoint_dir=checkpoint_dir)

    # Print final metrics
    if history["val_loss"]:
        best_epoch = int(np.argmin(history["val_loss"]))
        print(f"\nFinal results (best epoch {best_epoch + 1}):")
        print(f"  val_loss : {history['val_loss'][best_epoch]:.6f}")
        for k in ["delta", "xptar", "yptar", "ytar"]:
            rmse_key = f"val_rmse_{k}"
            if rmse_key in history:
                print(f"  {rmse_key} : {history[rmse_key][best_epoch]:.6f}")


if __name__ == "__main__":
    main()
