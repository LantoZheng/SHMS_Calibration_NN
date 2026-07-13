"""
FinetuneTrainer — Stage 2 fine-tuning loop on labeled sieve/foil data.

Loads a pre-trained checkpoint, freezes the backbone, and fine-tunes only
the output heads using a very small learning rate.  Optionally unfreezes
the full network after a configurable number of epochs for joint fine-tuning.
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


def _to_device(targets: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in targets.items()}


class FinetuneTrainer:
    """
    Stage-2 trainer: fine-tunes a pre-trained ResidualMLP on sieve/foil data.

    Parameters
    ----------
    model                 : ResidualMLP instance (architecture must match
                            the pre-trained checkpoint).
    loss_fn               : PhysicsInformedLoss instance.
    config                : dict loaded from finetune_config.yaml.
    pretrained_checkpoint : path to best_pretrain.pth.
    device                : 'cuda' or 'cpu'. Auto-detected if not specified.
    """

    def __init__(
        self,
        model: ResidualMLP,
        loss_fn: PhysicsInformedLoss,
        config: dict,
        pretrained_checkpoint: str,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.pretrained_checkpoint = pretrained_checkpoint
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def load_pretrained(self) -> None:
        """Load pre-trained weights and freeze the backbone."""
        ckpt = torch.load(self.pretrained_checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.freeze_backbone()
        print(
            f"Loaded pre-trained checkpoint from {self.pretrained_checkpoint} "
            f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.6f})"
        )

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: str = "checkpoints/finetune/",
    ) -> dict:
        """
        Run the fine-tuning loop.

        Returns
        -------
        dict  Training history.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        tcfg = self.config.get("training", {})
        epochs = tcfg.get("epochs", 300)
        batch_size = tcfg.get("batch_size", 256)
        lr = tcfg.get("learning_rate", 1e-5)
        weight_decay = tcfg.get("weight_decay", 1e-5)
        patience = tcfg.get("early_stopping_patience", 30)
        unfreeze_after = tcfg.get("unfreeze_after_epoch", None)
        unfreeze_lr = tcfg.get("unfreeze_lr", 1e-6)
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
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=0,
        )

        # ── Optimiser & scheduler ────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=True
        )

        best_val_loss = math.inf
        epochs_without_improvement = 0
        best_ckpt_path = os.path.join(checkpoint_dir, "best_finetune.pth")
        _unfrozen = False

        history: dict = {
            "train_loss": [],
            "val_loss": [],
            **{f"val_rmse_{k}": [] for k in _TARGET_KEYS},
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Optional unfreeze ─────────────────────────────────────────
            if (
                not _unfrozen
                and unfreeze_after is not None
                and epoch > unfreeze_after
            ):
                self.model.unfreeze_all()
                _unfrozen = True
                # Reset optimizer with very small LR for all parameters
                for pg in optimizer.param_groups:
                    pg["lr"] = unfreeze_lr
                optimizer.add_param_group(
                    {
                        "params": [
                            p for p in self.model.parameters()
                            if id(p) not in {id(q) for pg in optimizer.param_groups for q in pg["params"]}
                        ],
                        "lr": unfreeze_lr,
                        "weight_decay": weight_decay,
                    }
                )
                print(f"[Epoch {epoch}] Backbone unfrozen — lr={unfreeze_lr}")

            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            train_loss_accum = 0.0

            for batch in train_loader:
                inputs = batch["inputs"].to(self.device)
                targets = _to_device(batch["targets"], self.device)

                optimizer.zero_grad()
                inputs.requires_grad_(bool(self.loss_fn.transport_matrix))

                preds = self.model(inputs)
                loss = self.loss_fn(
                    preds, targets,
                    inputs=inputs if self.loss_fn.transport_matrix else None,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_accum += loss.item() * len(inputs)

            train_loss = train_loss_accum / len(train_dataset)

            # ── Validate ─────────────────────────────────────────────────
            val_loss, val_rmse = self._evaluate(val_loader)
            scheduler.step(val_loss)

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

        history_path = os.path.join(checkpoint_dir, "training_history_finetune.json")
        with open(history_path, "w") as fh:
            json.dump(history, fh, indent=2)

        print(f"\nBest val loss: {best_val_loss:.6f}  →  {best_ckpt_path}")
        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        sq_err: Dict[str, list] = {k: [] for k in _TARGET_KEYS}

        for batch in loader:
            inputs = batch["inputs"].to(self.device)
            targets = _to_device(batch["targets"], self.device)

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
