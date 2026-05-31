"""
Stage-2 trainer for ResidualTransportMLP on full ROOT weak labels.

Training phases
---------------
1. correction-head only
2. unfreeze correction branch, keep linear path fixed
3. optional full-model fine-tuning with very small learning rate
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from training.losses import Stage2WeakLabelLoss


_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]


class Stage2TransportTrainer:
    """Full-root stage-2 trainer aligned with the 5D transport backbone."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loss_fn: Stage2WeakLabelLoss,
        config: dict,
        pretrained_checkpoint: str,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.pretrained_checkpoint = pretrained_checkpoint
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self._print_device_summary()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        requested = str(device).strip().lower() if device is not None else None
        if requested:
            if requested.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was explicitly requested for Stage-2 training, but torch.cuda.is_available() is False.")
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _print_device_summary(self) -> None:
        print(f"Stage-2 trainer device: {self.device}")
        if self.device.type == "cuda":
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(device_index)
            gpu_name = torch.cuda.get_device_name(device_index)
            props = torch.cuda.get_device_properties(device_index)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(
                f"CUDA ready: index={device_index}, name={gpu_name}, capability={props.major}.{props.minor}, "
                f"total_memory={total_mem_gb:.1f} GB"
            )
            torch.backends.cudnn.benchmark = True

    def load_pretrained(self) -> None:
        ckpt = torch.load(self.pretrained_checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.freeze_backbone()
        first_param = next(self.model.parameters())
        print(
            f"Loaded stage-1 checkpoint from {self.pretrained_checkpoint} "
            f"(epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', float('nan')):.6f})"
        )
        print(f"Model parameter device after load: {first_param.device}")

    def _make_optimizer(self, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _split_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset, dict]:
        vcfg = self.config.get("validation", {})
        strategy = str(vcfg.get("strategy", "random"))
        seed = int(vcfg.get("random_seed", self.config.get("training", {}).get("random_seed", 42)))
        metadata = getattr(dataset, "metadata", None)

        if strategy == "random" or metadata is None:
            val_fraction = float(vcfg.get("val_fraction", self.config.get("training", {}).get("val_fraction", 0.2)))
            n_val = max(1, int(len(dataset) * val_fraction))
            n_train = len(dataset) - n_val
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
            return train_ds, val_ds, {"strategy": "random", "n_train": n_train, "n_val": n_val}

        if strategy == "leave_one_foil_out":
            foil_col = str(vcfg.get("foil_column", "foil_position"))
            holdout_foil = vcfg.get("holdout_foil")
            if holdout_foil is None:
                raise ValueError("validation.holdout_foil must be set for leave_one_foil_out.")
            val_mask = metadata[foil_col] == holdout_foil

        elif strategy == "leave_one_run_out":
            run_col = str(vcfg.get("run_column", "run_id"))
            holdout_run = vcfg.get("holdout_run")
            if holdout_run is None:
                raise ValueError("validation.holdout_run must be set for leave_one_run_out.")
            val_mask = metadata[run_col] == holdout_run

        elif strategy == "leave_some_holes_out":
            hole_col = str(vcfg.get("hole_column", "hole_id"))
            hole_fraction = float(vcfg.get("hole_fraction", 0.1))
            rng = np.random.default_rng(seed)
            unique_holes = metadata[hole_col].dropna().unique()
            n_holdout = max(1, int(len(unique_holes) * hole_fraction))
            holdout_holes = set(rng.choice(unique_holes, size=n_holdout, replace=False).tolist())
            val_mask = metadata[hole_col].isin(holdout_holes)
        else:
            raise ValueError(f"Unsupported validation strategy: {strategy}")

        train_idx = metadata.index[~val_mask].tolist()
        val_idx = metadata.index[val_mask].tolist()
        if not train_idx or not val_idx:
            raise RuntimeError(f"Validation split '{strategy}' produced an empty train or val set.")

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        summary = {
            "strategy": strategy,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }
        if strategy == "leave_one_foil_out":
            summary["holdout_foil"] = holdout_foil
        elif strategy == "leave_one_run_out":
            summary["holdout_run"] = holdout_run
        elif strategy == "leave_some_holes_out":
            summary["n_holdout_holes"] = n_holdout
        return train_ds, val_ds, summary

    def _switch_to_correction_branch_phase(self) -> None:
        if hasattr(self.model, "unfreeze_correction_branch"):
            self.model.unfreeze_correction_branch()
        if hasattr(self.model, "freeze_linear_path"):
            self.model.freeze_linear_path()

    def _switch_to_full_model_phase(self, keep_linear_path_frozen: bool) -> None:
        if hasattr(self.model, "unfreeze_all"):
            self.model.unfreeze_all()
        if keep_linear_path_frozen and hasattr(self.model, "freeze_linear_path"):
            self.model.freeze_linear_path()

    @staticmethod
    def _build_loader_kwargs(
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: Optional[int],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": max(int(num_workers), 0),
            "pin_memory": pin_memory,
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = bool(persistent_workers)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(prefetch_factor)
        return kwargs

    def train(
        self,
        *,
        dataset: Dataset,
        checkpoint_dir: str,
    ) -> dict:
        os.makedirs(checkpoint_dir, exist_ok=True)
        tcfg = self.config.get("training", {})
        epochs = int(tcfg.get("epochs", 120))
        batch_size = int(tcfg.get("batch_size", 1024))
        val_batch_size = int(tcfg.get("val_batch_size", batch_size * 2))
        weight_decay = float(tcfg.get("weight_decay", 1e-5))
        patience = int(tcfg.get("early_stopping_patience", 20))
        clip_norm = float(tcfg.get("gradient_clip_max_norm", 1.0))

        head_lr = float(tcfg.get("head_learning_rate", 2e-4))
        branch_unfreeze_epoch = int(tcfg.get("branch_unfreeze_epoch", 20))
        branch_lr = float(tcfg.get("branch_learning_rate", 5e-5))
        full_unfreeze_epoch = tcfg.get("full_unfreeze_epoch", None)
        full_lr = float(tcfg.get("full_learning_rate", 1e-5))
        full_model_disabled_targets = set(tcfg.get("full_model_disabled_targets", []))
        keep_linear_path_frozen = bool(tcfg.get("keep_linear_path_frozen", True))
        train_num_workers = int(tcfg.get("num_workers", 0))
        val_num_workers = int(tcfg.get("val_num_workers", train_num_workers))
        persistent_workers = bool(tcfg.get("persistent_workers", train_num_workers > 0))
        prefetch_factor = tcfg.get("prefetch_factor", None)

        train_ds, val_ds, split_summary = self._split_dataset(dataset)
        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            **self._build_loader_kwargs(
                batch_size=batch_size,
                shuffle=True,
                num_workers=train_num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            ),
        )
        val_loader = DataLoader(
            val_ds,
            **self._build_loader_kwargs(
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=val_num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            ),
        )

        optimizer = self._make_optimizer(head_lr, weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=8,
            factor=0.5,
        )

        best_val_loss = math.inf
        epochs_without_improvement = 0
        best_ckpt_path = os.path.join(checkpoint_dir, "best_finetune.pth")
        history: dict = {
            "train_loss": [],
            "val_loss": [],
            "split_summary": split_summary,
            **{f"val_center_rmse_{k}": [] for k in _TARGET_KEYS},
            **{f"val_deadzone_rmse_{k}": [] for k in _TARGET_KEYS},
            **{f"val_within_tol_{k}": [] for k in _TARGET_KEYS},
        }

        phase = "head_only"
        phase_switches = [{"epoch": 1, "phase": phase, "lr": head_lr}]
        pending_disable_targets: Optional[set] = None
        pending_disable_epoch: int = -1

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            if epoch == branch_unfreeze_epoch + 1 and phase == "head_only":
                self._switch_to_correction_branch_phase()
                optimizer = self._make_optimizer(branch_lr, weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=8,
                    factor=0.5,
                )
                phase = "correction_branch"
                phase_switches.append({"epoch": epoch, "phase": phase, "lr": branch_lr})
                disabled = set(tcfg.get("correction_branch_disabled_targets", []))
                delay = int(tcfg.get("correction_branch_disabled_targets_delay_epochs", 0))
                if disabled and delay > 0:
                    pending_disable_targets = disabled
                    pending_disable_epoch = epoch + delay
                    print(f"[Epoch {epoch}] Switched to correction-branch fine-tuning (lr={branch_lr})")
                    print(f"[Epoch {epoch}] {', '.join(sorted(disabled))} will be disabled at epoch {pending_disable_epoch}")
                elif disabled:
                    self.loss_fn.set_disabled_targets(disabled)
                    print(f"[Epoch {epoch}] Switched to correction-branch fine-tuning (lr={branch_lr})")
                else:
                    self.loss_fn.set_disabled_targets(set())
                    print(f"[Epoch {epoch}] Switched to correction-branch fine-tuning (lr={branch_lr})")

            if pending_disable_targets is not None and epoch >= pending_disable_epoch:
                self.loss_fn.set_disabled_targets(pending_disable_targets)
                pending_disable_targets = None

            if (
                full_unfreeze_epoch is not None
                and epoch == int(full_unfreeze_epoch) + 1
                and phase != "full_model"
            ):
                self._switch_to_full_model_phase(keep_linear_path_frozen=keep_linear_path_frozen)
                optimizer = self._make_optimizer(full_lr, weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=8,
                    factor=0.5,
                )
                phase = "full_model"
                phase_switches.append({"epoch": epoch, "phase": phase, "lr": full_lr})
                pending_disable_targets = None
                pending_disable_epoch = -1
                self.loss_fn.set_disabled_targets(full_model_disabled_targets)
                print(f"[Epoch {epoch}] Switched to full-model fine-tuning (lr={full_lr})")

            self.model.train()
            train_loss_accum = 0.0
            train_count = 0

            for batch in train_loader:
                inputs = batch["inputs"].to(self.device, non_blocking=(self.device.type == "cuda"))
                optimizer.zero_grad()
                preds = self.model(inputs)
                loss = self.loss_fn(preds, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                optimizer.step()

                train_loss_accum += loss.item() * len(inputs)
                train_count += len(inputs)

            train_loss = train_loss_accum / max(train_count, 1)
            val_loss, val_metrics = self._evaluate(val_loader)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for key in _TARGET_KEYS:
                history[f"val_center_rmse_{key}"].append(val_metrics.get(f"{key}_center_rmse", float("nan")))
                history[f"val_deadzone_rmse_{key}"].append(val_metrics.get(f"{key}_deadzone_rmse", float("nan")))
                history[f"val_within_tol_{key}"].append(val_metrics.get(f"{key}_within_tol", float("nan")))

            elapsed = time.time() - start_time
            metric_line = "  ".join(
                f"{key}_dead={val_metrics.get(f'{key}_deadzone_rmse', float('nan')):.4f}"
                for key in _TARGET_KEYS
            )
            print(
                f"[Epoch {epoch:>4}/{epochs}] phase={phase:<17} "
                f"train={train_loss:.5f}  val={val_loss:.5f}  {metric_line}  ({elapsed:.1f}s)"
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
                        "split_summary": split_summary,
                        "phase_switches": phase_switches,
                    },
                    best_ckpt_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        history["phase_switches"] = phase_switches
        with open(os.path.join(checkpoint_dir, "training_history_finetune.json"), "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(checkpoint_dir, "split_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(split_summary, fh, ensure_ascii=False, indent=2)

        print(f"\nBest val loss: {best_val_loss:.6f} -> {best_ckpt_path}")
        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        per_target: Dict[str, list[float]] = {f"{k}_{m}": [] for k in _TARGET_KEYS for m in ["center_rmse", "deadzone_rmse", "within_tol"]}

        for batch in loader:
            inputs = batch["inputs"].to(self.device, non_blocking=(self.device.type == "cuda"))
            preds = self.model(inputs)
            loss = self.loss_fn(preds, batch)
            total_loss += loss.item() * len(inputs)
            total_count += len(inputs)

            metrics = self.loss_fn.compute_metrics(preds, batch)
            for key, value in metrics.items():
                if not math.isnan(value):
                    per_target.setdefault(key, []).append(value)

        val_loss = total_loss / max(total_count, 1)
        aggregated = {
            key: float(np.mean(values)) if values else float("nan")
            for key, values in per_target.items()
        }
        return val_loss, aggregated
