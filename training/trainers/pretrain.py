"""
PretrainTrainer — Stage 1 training loop on SIMC Monte-Carlo data.

Trains ResidualMLP with PhysicsInformedLoss using AdamW + CosineAnnealingLR.
Supports mixed-precision training on CUDA, gradient clipping, and
early stopping.  Saves best checkpoint and training history JSON.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from training.models.physics_loss import PhysicsInformedLoss
from training.models.residual_mlp import ResidualMLP

_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]


def _dict_targets_to_device(
    targets: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in targets.items()}


class PretrainTrainer:
    """
    Stage-1 trainer: pre-trains ResidualMLP on SIMC simulation data.

    Parameters
    ----------
    model   : ResidualMLP instance (freshly initialised).
    loss_fn : PhysicsInformedLoss instance.
    config  : dict loaded from pretrain_config.yaml (or equivalent).
    device  : 'cuda' or 'cpu'. Auto-detected if not specified.
    """

    def __init__(
        self,
        model: ResidualMLP,
        loss_fn: PhysicsInformedLoss,
        config: dict,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: str = "checkpoints/pretrain/",
    ) -> dict:
        """
        Run the pre-training loop.

        Parameters
        ----------
        train_dataset  : SIMCDataset (or any Dataset returning the expected dict).
        val_dataset    : validation dataset; if None, split is taken from
                         config['training']['val_fraction'].
        checkpoint_dir : where to write checkpoints and history JSON.

        Returns
        -------
        dict  Training history with keys 'train_loss', 'val_loss',
              and per-target RMSE lists.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        tcfg = self.config.get("training", {})
        epochs = tcfg.get("epochs", 200)
        batch_size = tcfg.get("batch_size", 2048)
        lr = tcfg.get("learning_rate", 1e-3)
        weight_decay = tcfg.get("weight_decay", 1e-4)
        patience = tcfg.get("early_stopping_patience", 20)
        grad_clip = tcfg.get("gradient_clip_max_norm", 1.0)
        val_fraction = tcfg.get("val_fraction", 0.2)
        seed = tcfg.get("random_seed", 42)

        # ── Train / val split ────────────────────────────────────────────
        if val_dataset is None:
            n_val = max(1, int(len(train_dataset) * val_fraction))
            n_train = len(train_dataset) - n_val
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = random_split(
                train_dataset, [n_train, n_val], generator=generator
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
        )

        # ── Optimiser & scheduler ────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        use_amp = self.device.type == "cuda"
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ── Training state ───────────────────────────────────────────────
        best_val_loss = math.inf
        epochs_without_improvement = 0
        best_ckpt_path = os.path.join(checkpoint_dir, "best_pretrain.pth")

        history: dict = {
            "train_loss": [],
            "val_loss": [],
            **{f"val_rmse_{k}": [] for k in _TARGET_KEYS},
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            train_loss_accum = 0.0

            for batch in train_loader:
                inputs = batch["inputs"].to(self.device)
                targets = _dict_targets_to_device(batch["targets"], self.device)

                optimizer.zero_grad()

                # Enable grad on inputs for physics penalty
                inputs.requires_grad_(bool(self.loss_fn.transport_matrix))

                with torch.cuda.amp.autocast(enabled=use_amp):
                    preds = self.model(inputs)
                    loss = self.loss_fn(
                        preds, targets,
                        inputs=inputs if self.loss_fn.transport_matrix else None,
                    )

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()

                train_loss_accum += loss.item() * len(inputs)

            train_loss = train_loss_accum / len(train_dataset)

            # ── Validate ─────────────────────────────────────────────────
            val_loss, val_rmse = self._evaluate(val_loader)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for k in _TARGET_KEYS:
                history[f"val_rmse_{k}"].append(val_rmse.get(k, float("nan")))

            elapsed = time.time() - t0
            rmse_str = "  ".join(
                f"{k}={val_rmse.get(k, float('nan')):.4f}" for k in _TARGET_KEYS
            )
            print(
                f"[Epoch {epoch:>4}/{epochs}]  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"{rmse_str}  ({elapsed:.1f}s)"
            )

            # ── Early stopping & checkpoint ───────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": self.config,
                    },
                    best_ckpt_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)."
                    )
                    break

        # ── Save history ─────────────────────────────────────────────────
        history_path = os.path.join(checkpoint_dir, "training_history_pretrain.json")
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        print(f"\nBest val loss: {best_val_loss:.6f}  →  {best_ckpt_path}")
        return history

    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader
    ) -> tuple:
        """Return (val_loss, {target: rmse}) over the loader."""
        self.model.eval()
        total_loss = 0.0
        sq_err: Dict[str, list] = {k: [] for k in _TARGET_KEYS}

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            targets = _dict_targets_to_device(batch["targets"], self.device)

            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets, inputs=None)
            total_loss += loss.item() * len(inputs)

            for k in _TARGET_KEYS:
                if k in preds and k in targets:
                    err = (preds[k] - targets[k]).squeeze().cpu().numpy()
                    sq_err[k].extend((err ** 2).tolist())

        n = sum(len(v) for v in sq_err.values()) // max(1, len(_TARGET_KEYS))
        val_loss = total_loss / max(n, 1)
        val_rmse = {
            k: float(np.sqrt(np.mean(v))) if v else float("nan")
            for k, v in sq_err.items()
        }
        return val_loss, val_rmse
