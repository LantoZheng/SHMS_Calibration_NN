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
    from training.models import build_model_from_config
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
    output_cfg = config.setdefault("output", {})
    original_checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints/pretrain/")
    if args.output_dir:
        output_cfg["checkpoint_dir"] = args.output_dir
    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints/pretrain/")

    configured_scaler_path = output_cfg.get("scaler_save_path")
    if configured_scaler_path:
        configured_scaler_path_norm = os.path.normpath(configured_scaler_path)
        original_default_scaler = os.path.normpath(os.path.join(original_checkpoint_dir, "scaler_bundle.json"))
        if args.output_dir and configured_scaler_path_norm == original_default_scaler:
            scaler_path = os.path.join(checkpoint_dir, "scaler_bundle.json")
        else:
            scaler_path = configured_scaler_path
    else:
        scaler_path = os.path.join(checkpoint_dir, "scaler_bundle.json")

    p0 = args.p0 if args.p0 is not None else None
    tree_name = config.get("data", {}).get("simc_tree_name", "h10")
    include_fry = config.get("data", {}).get("include_fry", False)
    include_xtar = config.get("data", {}).get("include_xtar", False)
    include_p0 = config.get("data", {}).get("include_p0", False)
    fry_branch = config.get("data", {}).get("fry_branch", None)
    x_tar_sigma = config.get("data", {}).get("x_tar_sigma_cm", 0.1)
    feature_schema = config.get("data", {}).get("feature_schema", None)
    if feature_schema is None:
        feature_schema = ["x_fp", "y_fp", "xp_fp", "yp_fp"]
        if include_fry:
            feature_schema.append("fry")
        if include_xtar:
            feature_schema.append("x_tar")
        if include_p0:
            feature_schema.append("p0")

    # Build dataset — fit scalers on this data
    print("Loading SIMC data …")
    dataset = SIMCDataset(
        root_file_paths=root_files,
        tree_name=tree_name,
        p0_value=p0,
        max_events=args.max_events,
        fit_scalers=True,
        feature_schema=feature_schema,
        include_fry=include_fry,
        fry_branch=fry_branch,
        x_tar_sigma_cm=x_tar_sigma,
        rng_seed=config.get("training", {}).get("random_seed", 42),
    )
    print(f"Dataset size: {len(dataset)} events")

    # Save scaler bundle
    mcfg = config.get("model", {})
    input_features = list(dataset.feature_names)
    target_features = ["delta", "xptar", "yptar", "ytar"]
    bundle = ScalerBundle(input_features=input_features, target_features=target_features)
    if dataset.scaler_X is not None and dataset.scaler_Y is not None:
        bundle.set_fitted_scalers(dataset.scaler_X, dataset.scaler_Y)
        bundle.save(scaler_path)
        print(f"Scaler bundle saved to: {scaler_path}")

    # Build model
    input_dim = len(input_features)
    if mcfg.get("input_dim", input_dim) != input_dim:
        print(
            f"Warning: config model.input_dim={mcfg.get('input_dim')} does not match derived input_dim={input_dim}; using derived value."
        )
    model = build_model_from_config(mcfg, input_dim=input_dim)
    model.model_summary()

    # Build loss
    lcfg = config.get("loss", {})
    transport_matrix = lcfg.get("transport_matrix", None)
    loss_fn = PhysicsInformedLoss(
        lambda_physics=lcfg.get("lambda_physics", 0.01),
        use_huber=lcfg.get("use_huber", True),
        huber_delta=lcfg.get("huber_delta", 1.0),
        transport_matrix=transport_matrix if transport_matrix else None,
        target_weights=lcfg.get("target_weights", None),
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
