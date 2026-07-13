"""
PretrainTrainer — Stage 1 training loop on SIMC Monte-Carlo data.

Supports both the baseline `ResidualMLP` and the structured
`ResidualTransportMLP`, including optional least-squares linear initialisation,
linear warmup, and correction-branch regularisation.
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
        early_stopping_min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
        grad_clip = tcfg.get("gradient_clip_max_norm", 1.0)
        val_fraction = tcfg.get("val_fraction", 0.2)
        seed = tcfg.get("random_seed", 42)
        scheduler_cfg = dict(tcfg.get("scheduler", {}))
        mcfg = self.config.get("model", {})
        linear_warmup_epochs = int(mcfg.get("linear_warmup_epochs", 0))
        freeze_linear_after_warmup = bool(mcfg.get("freeze_linear_after_warmup", False))
        init_linear_with_least_squares = bool(mcfg.get("init_linear_with_least_squares", False))
        linear_init_ridge = float(mcfg.get("linear_init_ridge", 1e-6))
        correction_l2_weight = float(mcfg.get("correction_l2_weight", 0.0))

        # ── Train / val split ────────────────────────────────────────────
        if val_dataset is None:
            n_val = max(1, int(len(train_dataset) * val_fraction))
            n_train = len(train_dataset) - n_val
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = random_split(
                train_dataset, [n_train, n_val], generator=generator
            )

        if init_linear_with_least_squares and hasattr(self.model, "initialise_linear_path_least_squares"):
            source_dataset = getattr(train_dataset, "dataset", train_dataset)
            indices = getattr(train_dataset, "indices", None)
            X_train = source_dataset.X if indices is None else source_dataset.X[indices]
            Y_train = source_dataset.Y if indices is None else source_dataset.Y[indices]
            self.model.initialise_linear_path_least_squares(X_train, Y_train, ridge=linear_init_ridge)
            print(f"Initialised linear transport path with least squares (ridge={linear_init_ridge:g}).")

        if linear_warmup_epochs > 0 and hasattr(self.model, "freeze_correction_branch"):
            self.model.freeze_correction_branch()
            print(f"Linear warmup enabled for first {linear_warmup_epochs} epoch(s).")

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
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = self._build_scheduler(optimizer, scheduler_cfg, epochs)
        scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()

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
        warmup_finished = linear_warmup_epochs <= 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            if (
                not warmup_finished
                and epoch > linear_warmup_epochs
                and hasattr(self.model, "unfreeze_correction_branch")
            ):
                self.model.unfreeze_correction_branch()
                if freeze_linear_after_warmup and hasattr(self.model, "freeze_linear_path"):
                    self.model.freeze_linear_path()
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                scheduler = self._build_scheduler(
                    optimizer,
                    scheduler_cfg,
                    max(epochs - epoch + 1, 1),
                )
                warmup_finished = True
                print(f"[Epoch {epoch}] Linear warmup finished; correction branch unfrozen.")

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
                    if correction_l2_weight > 0.0 and "correction" in preds:
                        loss = loss + correction_l2_weight * torch.mean(preds["correction"] ** 2)

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()

                train_loss_accum += loss.item() * len(inputs)

            train_loss = train_loss_accum / len(train_dataset)

            # ── Validate ─────────────────────────────────────────────────
            val_loss, val_rmse = self._evaluate(val_loader)

            if scheduler_type in {"plateau", "reduce_on_plateau"}:
                scheduler.step(val_loss)
            else:
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
            if val_loss < (best_val_loss - early_stopping_min_delta):
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

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: dict,
        epochs: int,
    ):
        scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
        if scheduler_type in {"plateau", "reduce_on_plateau"}:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 5)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
        )

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
