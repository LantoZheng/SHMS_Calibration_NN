#!/usr/bin/env python3
"""
Stage-2 fine-tuning entry point for the 5D ResidualTransportMLP full-ROOT workflow.

This script is intended for the upcoming SHMS `3foils + sieve` full ROOT file.
It assumes that the weak-label construction pipeline has already produced
geometric supervision columns inside the tree (or an equivalent CSV / Parquet
export).
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 full-ROOT fine-tuning for ResidualTransportMLP")
    parser.add_argument("--config", required=True, help="Path to stage2 YAML config")
    parser.add_argument("--root-file", required=True, help="Path to labeled full ROOT / CSV / Parquet file")
    parser.add_argument("--tree-name", default=None, help="Override ROOT tree name")
    parser.add_argument("--checkpoint", default=None, help="Override stage-1 pretrained checkpoint")
    parser.add_argument("--scaler-bundle", default=None, help="Override stage-1 scaler bundle")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint output directory")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap for smoke tests")
    return parser.parse_args()


def main() -> None:
    import torch
    from training.data.preprocessing import ScalerBundle
    from training.data.stage2_root_dataset import Stage2RootDataset
    from training.losses import Stage2WeakLabelLoss
    from training.models import build_model_from_config
    from training.trainers.stage2_transport import Stage2TransportTrainer

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    pretrained_ckpt = args.checkpoint or config["pretrained"]["checkpoint_path"]
    scaler_path = args.scaler_bundle or config["pretrained"]["scaler_bundle_path"]
    tree_name = args.tree_name or config.get("data", {}).get("tree_name", "T")
    output_dir = args.output_dir or config["output"]["checkpoint_dir"]

    print(f"Requested device argument: {args.device or 'auto'}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Loading scaler bundle from: {scaler_path}")
    scaler_bundle = ScalerBundle.load(scaler_path)

    dcfg = config.get("data", {})
    dataset = Stage2RootDataset(
        data_source=args.root_file,
        tree_name=tree_name,
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
        max_events=args.max_events,
    )
    print(f"Stage-2 dataset: raw={dataset.summary.raw_events}, kept={dataset.summary.kept_events}")
    print(f"Cutflow: {dataset.summary.cutflow}")

    model_cfg = dict(config.get("model", {}))
    model = build_model_from_config(model_cfg, input_dim=dataset.X.shape[1])

    loss_cfg = dict(config.get("loss", {}))
    loss_fn = Stage2WeakLabelLoss(
        use_huber=loss_cfg.get("use_huber", True),
        huber_delta=loss_cfg.get("huber_delta", 1.0),
        target_weights=loss_cfg.get("target_weights", {}),
        correction_l2_weight=loss_cfg.get("correction_l2_weight", 0.0),
    )

    trainer = Stage2TransportTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        pretrained_checkpoint=pretrained_ckpt,
        device=args.device,
    )
    trainer.load_pretrained()
    trainer.train(dataset=dataset, checkpoint_dir=output_dir)


if __name__ == "__main__":
    main()
