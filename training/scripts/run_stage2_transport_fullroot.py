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

import numpy as np
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
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip automatic post-training reconstruction visuals")
    parser.add_argument("--postprocess-output-dir", default=None, help="Override post-training visualization output directory")
    return parser.parse_args()


def build_mechanical_hole_center_bank(dataset, scaler_bundle, data_cfg: dict) -> dict[int, dict[str, list[list[float]]]]:
    label_map = dict(data_cfg.get("label_map", {}))
    metadata_cols = dict(data_cfg.get("metadata_cols", {}))
    foil_col = str(metadata_cols.get("foil_position", "foil_position"))
    row_col = str(metadata_cols.get("hole_row", "hole_row"))
    col_col = str(metadata_cols.get("hole_col", "hole_col"))
    x_center_col = str(label_map.get("xptar", {}).get("center_col", "weak_hole_xptar_center"))
    y_center_col = str(label_map.get("yptar", {}).get("center_col", "weak_hole_yptar_center"))

    required_cols = [foil_col, row_col, col_col, x_center_col, y_center_col]
    df = dataset.df[required_cols].dropna().copy()
    if df.empty:
        return {}

    df[foil_col] = df[foil_col].astype(int)
    df[row_col] = df[row_col].astype(int)
    df[col_col] = df[col_col].astype(int)
    grouped = (
        df.groupby([foil_col, row_col, col_col], as_index=False)[[x_center_col, y_center_col]]
        .median()
        .sort_values([foil_col, row_col, col_col])
    )

    center_xy = grouped[[x_center_col, y_center_col]].to_numpy(dtype=np.float32, copy=True)
    if scaler_bundle is not None:
        center_xy[:, 0] = (center_xy[:, 0] - float(scaler_bundle.scaler_Y.mean_[1])) / float(scaler_bundle.scaler_Y.scale_[1])
        center_xy[:, 1] = (center_xy[:, 1] - float(scaler_bundle.scaler_Y.mean_[2])) / float(scaler_bundle.scaler_Y.scale_[2])

    bank: dict[int, dict[str, list[list[float]]]] = {}
    for foil_value, sub in grouped.groupby(foil_col, sort=True):
        foil_int = int(foil_value)
        bank[foil_int] = {
            "keys": sub[[row_col, col_col]].to_numpy(dtype=np.int64).tolist(),
            "centers": center_xy[sub.index.to_numpy()].tolist(),
        }
    return bank


def main() -> None:
    import torch
    from training.data.preprocessing import ScalerBundle
    from training.data.stage2_root_dataset import Stage2RootDataset
    from training.losses import Stage2WeakLabelLoss
    from training.models import build_model_from_config
    from training.scripts.analyze_stage2_reconstruction_visuals import run_post_training_visuals
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
    hole_center_bank = build_mechanical_hole_center_bank(dataset, scaler_bundle, dcfg)
    if hole_center_bank:
        print(
            "Mechanical hole center bank: "
            + ", ".join(f"foil {foil}: {len(payload['keys'])} holes" for foil, payload in sorted(hole_center_bank.items()))
        )

    model_cfg = dict(config.get("model", {}))
    model = build_model_from_config(model_cfg, input_dim=dataset.X.shape[1])

    tcfg = dict(config.get("training", {}))
    loss_cfg = dict(config.get("loss", {}))
    loss_fn = Stage2WeakLabelLoss(
        use_huber=loss_cfg.get("use_huber", True),
        huber_delta=loss_cfg.get("huber_delta", 1.0),
        target_weights=loss_cfg.get("target_weights", {}),
        correction_l2_weight=loss_cfg.get("correction_l2_weight", 0.0),
        hole_separation_weight=loss_cfg.get("hole_separation_weight", 0.0),
        hole_separation_temperature=loss_cfg.get("hole_separation_temperature", 0.5),
        hole_center_bank=hole_center_bank,
        sieve_plane_weight=loss_cfg.get("sieve_plane_weight", 0.0),
        sieve_plane_huber_delta_cm=loss_cfg.get("sieve_plane_huber_delta_cm", 0.3),
        sieve_distance_cm=loss_cfg.get("sieve_distance_cm", 253.0),
        target_scales={
            "delta": float(scaler_bundle.scaler_Y.scale_[0]),
            "xptar": float(scaler_bundle.scaler_Y.scale_[1]),
            "yptar": float(scaler_bundle.scaler_Y.scale_[2]),
            "ytar": float(scaler_bundle.scaler_Y.scale_[3]),
        },
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

    if not args.skip_postprocess:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir_abs = output_dir if os.path.isabs(output_dir) else os.path.join(repo_root, output_dir)
        checkpoint_path = os.path.join(output_dir_abs, "best_finetune.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expected best checkpoint not found after training: {checkpoint_path}")

        pcfg = dict(config.get("postprocess", {}))
        post_dir = args.postprocess_output_dir or pcfg.get("output_dir")
        if not post_dir:
            ckpt_name = os.path.basename(os.path.normpath(output_dir_abs))
            post_dir = os.path.join(repo_root, "outputs", f"{ckpt_name}_postprocess")

        print("\nRunning automatic post-training reconstruction visuals...")
        payload = run_post_training_visuals(
            checkpoint=checkpoint_path,
            data=args.root_file,
            output_dir=post_dir,
            scaler_bundle=scaler_path,
            device=args.device,
            batch_size=int(pcfg.get("batch_size", tcfg.get("val_batch_size", tcfg.get("batch_size", 4096)))),
            max_events=args.max_events,
            sieve_distance_cm=float(pcfg.get("sieve_distance_cm", loss_cfg.get("sieve_distance_cm", 253.0))),
        )
        print("Post-training visuals complete:")
        for name, artifact in payload.get("artifacts", {}).items():
            print(f"  {name}: {artifact}")


if __name__ == "__main__":
    main()
