#!/usr/bin/env python3
"""
Stage 2 — Sieve/foil fine-tuning entry point.

Usage
-----
python training/scripts/run_finetune.py \\
    --config training/configs/finetune_config.yaml \\
    --sieve-data /path/to/labeled_sieve_data.csv \\
    --pretrained-checkpoint checkpoints/pretrain/best_pretrain.pth \\
    --p0 4.4 \\
    [--device cuda]

The sieve data must be a CSV or Parquet file whose column names follow
the hcana convention (P_dc_x_fp, P_gtr_dp, etc.) as produced by
SHMS_Optics_calibration_tools.

The scaler bundle from pre-training is reused automatically.
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHMS optics — Stage 2 sieve/foil fine-tuning")
    parser.add_argument("--config", required=True, help="Path to finetune_config.yaml")
    parser.add_argument(
        "--sieve-data",
        required=True,
        help="Path to labeled sieve/foil data (CSV or Parquet)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Override pretrained checkpoint path from config",
    )
    parser.add_argument(
        "--scaler-bundle",
        default=None,
        help="Path to ScalerBundle JSON from pre-training (overrides config)",
    )
    parser.add_argument("--p0", type=float, default=None, help="Central momentum GeV/c")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint output directory")
    return parser.parse_args()


def main() -> None:
    import yaml
    from training.data.sieve_dataset import SieveDataset
    from training.data.preprocessing import ScalerBundle
    from training.models.residual_mlp import ResidualMLP
    from training.models.physics_loss import PhysicsInformedLoss
    from training.trainers.finetune import FinetuneTrainer

    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # Resolve checkpoint / scaler paths
    pretrained_ckpt = (
        args.pretrained_checkpoint
        or config.get("pretrained", {}).get("checkpoint_path", "checkpoints/pretrain/best_pretrain.pth")
    )
    scaler_path = (
        args.scaler_bundle
        or config.get("output", {}).get("scaler_save_path", "checkpoints/pretrain/scaler_bundle.json")
    )
    checkpoint_dir = (
        args.output_dir
        or config.get("output", {}).get("checkpoint_dir", "checkpoints/finetune/")
    )

    p0 = args.p0 if args.p0 is not None else config.get("data", {}).get("p0_value", None)

    # Load scaler bundle from pre-training (MUST reuse — do not refit)
    print(f"Loading scaler bundle from: {scaler_path}")
    scaler_bundle = ScalerBundle.load(scaler_path)

    # Build dataset
    dcfg = config.get("data", {})
    x_tar_col = dcfg.get("x_tar_col", "P_react_x")
    weight_col = dcfg.get("weight_col", None)

    print(f"Loading sieve data from: {args.sieve_data}")
    dataset = SieveDataset(
        data_source=args.sieve_data,
        p0_value=p0,
        x_tar_col=x_tar_col,
        weight_col=weight_col,
        scaler_X=scaler_bundle.scaler_X,
        scaler_Y=scaler_bundle.scaler_Y,
    )
    print(f"Dataset size: {len(dataset)} events")

    # Build model (architecture must match pre-training)
    mcfg = config.get("model", {})
    model = ResidualMLP(
        input_dim=mcfg.get("input_dim", 6),
        hidden_dim=mcfg.get("hidden_dim", 256),
        n_residual_blocks=mcfg.get("n_residual_blocks", 4),
        branch_dim=mcfg.get("branch_dim", 64),
    )

    # Build loss
    lcfg = config.get("loss", {})
    loss_fn = PhysicsInformedLoss(
        lambda_physics=lcfg.get("lambda_physics", 0.005),
        use_huber=lcfg.get("use_huber", True),
        huber_delta=lcfg.get("huber_delta", 1.0),
    )

    # Fine-tune
    trainer = FinetuneTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        pretrained_checkpoint=pretrained_ckpt,
        device=args.device,
    )
    trainer.load_pretrained()

    history = trainer.train(train_dataset=dataset, checkpoint_dir=checkpoint_dir)

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
