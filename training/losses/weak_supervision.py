"""
Weak-label losses for stage-2 SHMS fine-tuning.

The full-root stage-2 workflow supervises the model with geometric labels:

- sieve-hole labels provide centre + tolerance for xptar / yptar
- ytar may be either a strong foil-centre target (zero tolerance) or a
    tolerance-aware label, depending on the dataset/configuration
- delta can optionally remain a point target with zero tolerance

The loss used here is a dead-zone Huber loss:

    excess = max(|prediction - centre| - tolerance, 0)
    loss   = Huber(excess, 0)

Inside the known physical tolerance window, the sample contributes zero loss.
Outside the window, the penalty grows smoothly and remains robust to outliers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]


class Stage2WeakLabelLoss(nn.Module):
    """Dead-zone Huber loss for stage-2 weak supervision."""

    def __init__(
        self,
        *,
        use_huber: bool = True,
        huber_delta: float = 1.0,
        target_weights: Optional[Dict[str, float]] = None,
        correction_l2_weight: float = 0.0,
        hole_separation_weight: float = 0.0,
        hole_separation_temperature: float = 0.5,
        hole_center_bank: Optional[Dict[int, Dict[str, Any]]] = None,
        sieve_plane_weight: float = 0.0,
        sieve_plane_huber_delta_cm: float = 0.3,
        sieve_distance_cm: float = 253.0,
        target_scales: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)
        self.target_weights = {
            key: float((target_weights or {}).get(key, 1.0))
            for key in _TARGET_KEYS
        }
        self.correction_l2_weight = float(correction_l2_weight)
        self.hole_separation_weight = float(hole_separation_weight)
        self.hole_separation_temperature = float(max(hole_separation_temperature, 1e-6))
        self.hole_center_bank = self._normalize_hole_center_bank(hole_center_bank)
        self.sieve_plane_weight = float(max(sieve_plane_weight, 0.0))
        self.sieve_plane_huber_delta_cm = float(max(sieve_plane_huber_delta_cm, 1e-6))
        self.sieve_distance_cm = float(max(sieve_distance_cm, 1e-6))
        self.target_scales = {
            key: float((target_scales or {}).get(key, 1.0))
            for key in _TARGET_KEYS
        }
        self._disabled_targets: Set[str] = set()

    @staticmethod
    def _normalize_hole_center_bank(
        hole_center_bank: Optional[Dict[int, Dict[str, Any]]],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        normalized: Dict[int, Dict[str, torch.Tensor]] = {}
        if not hole_center_bank:
            return normalized

        for foil, payload in hole_center_bank.items():
            keys = torch.as_tensor(payload["keys"], dtype=torch.long)
            centers = torch.as_tensor(payload["centers"], dtype=torch.float32)
            if keys.ndim != 2 or keys.shape[1] != 2:
                raise ValueError(f"hole_center_bank foil {foil} keys must have shape [N, 2].")
            if centers.ndim != 2 or centers.shape[1] != 2:
                raise ValueError(f"hole_center_bank foil {foil} centers must have shape [N, 2].")
            if keys.shape[0] != centers.shape[0]:
                raise ValueError(f"hole_center_bank foil {foil} keys/centers size mismatch.")
            normalized[int(foil)] = {
                "keys": keys,
                "centers": centers,
            }
        return normalized

    def set_disabled_targets(self, targets: Optional[Set[str]] = None) -> None:
        self._disabled_targets = set(targets or [])
        if self._disabled_targets:
            disabled_str = ", ".join(sorted(self._disabled_targets))
            print(f"[loss] disabled targets: {disabled_str}")

    def _deadzone_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tolerance: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        err = torch.abs(pred - target)
        excess = torch.clamp(err - tolerance, min=0.0)
        zeros = torch.zeros_like(excess)
        if self.use_huber:
            per_sample = F.huber_loss(
                excess,
                zeros,
                delta=self.huber_delta,
                reduction="none",
            )
        else:
            per_sample = excess.square()

        weighted = per_sample * weight
        denom = torch.clamp(weight.sum(), min=1e-12)
        return weighted.sum() / denom

    def _hole_separation_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, object],
        sample_weight: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if self.hole_separation_weight <= 0.0 or not self.hole_center_bank:
            return torch.tensor(0.0, device=device)
        if "xptar" not in predictions or "yptar" not in predictions:
            return torch.tensor(0.0, device=device)

        metadata = batch.get("metadata", None)
        if not isinstance(metadata, dict):
            return torch.tensor(0.0, device=device)
        if not all(key in metadata for key in ["foil_position", "hole_row", "hole_col"]):
            return torch.tensor(0.0, device=device)

        x_mask = batch["target_mask"]["xptar"].to(device).squeeze(-1) > 0.5
        y_mask = batch["target_mask"]["yptar"].to(device).squeeze(-1) > 0.5
        valid_mask = x_mask & y_mask
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=device)

        foil_position = torch.as_tensor(metadata["foil_position"], device=device).reshape(-1).long()
        hole_row = torch.as_tensor(metadata["hole_row"], device=device).reshape(-1).long()
        hole_col = torch.as_tensor(metadata["hole_col"], device=device).reshape(-1).long()

        pred_xy = torch.stack(
            [predictions["xptar"].squeeze(-1), predictions["yptar"].squeeze(-1)],
            dim=1,
        )
        if sample_weight is None:
            sample_weight = torch.ones(pred_xy.shape[0], device=device, dtype=torch.float32)
        else:
            sample_weight = sample_weight.to(device).reshape(-1).float()

        weighted_sum = torch.tensor(0.0, device=device)
        weight_denom = torch.tensor(0.0, device=device)

        for foil_value in torch.unique(foil_position[valid_mask]).tolist():
            foil_int = int(foil_value)
            if foil_int not in self.hole_center_bank:
                continue
            foil_mask = valid_mask & (foil_position == foil_int)
            if not torch.any(foil_mask):
                continue

            bank = self.hole_center_bank[foil_int]
            bank_keys = bank["keys"].to(device)
            bank_centers = bank["centers"].to(device)
            pred_f = pred_xy[foil_mask]
            row_f = hole_row[foil_mask]
            col_f = hole_col[foil_mask]
            weight_f = sample_weight[foil_mask]

            matches = (
                (bank_keys[:, 0].unsqueeze(0) == row_f.unsqueeze(1))
                & (bank_keys[:, 1].unsqueeze(0) == col_f.unsqueeze(1))
            )
            has_target = matches.any(dim=1)
            if not torch.any(has_target):
                continue

            pred_f = pred_f[has_target]
            weight_f = weight_f[has_target]
            matches = matches[has_target]
            target_index = matches.float().argmax(dim=1)
            logits = -torch.sum((pred_f.unsqueeze(1) - bank_centers.unsqueeze(0)).square(), dim=2) / self.hole_separation_temperature
            per_sample = F.cross_entropy(logits, target_index, reduction="none")
            weighted_sum = weighted_sum + torch.sum(per_sample * weight_f)
            weight_denom = weight_denom + torch.sum(weight_f)

        if weight_denom.item() <= 0:
            return torch.tensor(0.0, device=device)
        return weighted_sum / torch.clamp(weight_denom, min=1e-12)

    def _sieve_plane_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, object],
        sample_weight: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if self.sieve_plane_weight <= 0.0:
            return torch.tensor(0.0, device=device)
        if "xptar" not in predictions or "yptar" not in predictions:
            return torch.tensor(0.0, device=device)

        targets = batch["targets"]
        tolerances = batch["tolerances"]
        masks = batch["target_mask"]

        x_pred = predictions["xptar"].squeeze(-1)
        y_pred = predictions["yptar"].squeeze(-1)
        x_tgt = targets["xptar"].to(device).squeeze(-1)
        y_tgt = targets["yptar"].to(device).squeeze(-1)
        x_tol = tolerances["xptar"].to(device).squeeze(-1).clamp_min(0.0)
        y_tol = tolerances["yptar"].to(device).squeeze(-1).clamp_min(0.0)
        x_mask = masks["xptar"].to(device).squeeze(-1) > 0.5
        y_mask = masks["yptar"].to(device).squeeze(-1) > 0.5
        valid_mask = x_mask & y_mask
        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=device)

        if sample_weight is None:
            w = torch.ones_like(x_pred, device=device)
        else:
            w = sample_weight.to(device).reshape(-1).float()

        x_excess_scaled = torch.clamp(torch.abs(x_pred - x_tgt) - x_tol, min=0.0)
        y_excess_scaled = torch.clamp(torch.abs(y_pred - y_tgt) - y_tol, min=0.0)

        x_excess_cm = x_excess_scaled * (self.target_scales.get("xptar", 1.0) * self.sieve_distance_cm)
        y_excess_cm = y_excess_scaled * (self.target_scales.get("yptar", 1.0) * self.sieve_distance_cm)
        radial_excess_cm = torch.sqrt(x_excess_cm.square() + y_excess_cm.square())
        radial_excess_cm = radial_excess_cm[valid_mask]
        w_valid = w[valid_mask]

        zeros = torch.zeros_like(radial_excess_cm)
        if self.use_huber:
            per_sample = F.huber_loss(
                radial_excess_cm,
                zeros,
                delta=self.sieve_plane_huber_delta_cm,
                reduction="none",
            )
        else:
            per_sample = radial_excess_cm.square()

        denom = torch.clamp(w_valid.sum(), min=1e-12)
        return torch.sum(per_sample * w_valid) / denom

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, object]) -> torch.Tensor:
        targets = batch["targets"]
        tolerances = batch["tolerances"]
        masks = batch["target_mask"]
        sample_weight = batch.get("weight", None)

        device = next(iter(predictions.values())).device
        total = torch.tensor(0.0, device=device)

        for key in _TARGET_KEYS:
            if key in self._disabled_targets:
                continue
            if key not in predictions:
                continue

            pred = predictions[key].squeeze(-1)
            tgt = targets[key].to(device).squeeze(-1)
            tol = tolerances[key].to(device).squeeze(-1).clamp_min(0.0)
            mask = masks[key].to(device).squeeze(-1) > 0.5
            if mask.sum().item() == 0:
                continue

            if sample_weight is None:
                w = torch.ones_like(pred, device=device)
            else:
                w = sample_weight.to(device).reshape(-1)

            loss_k = self._deadzone_loss(
                pred=pred[mask],
                target=tgt[mask],
                tolerance=tol[mask],
                weight=w[mask],
            )
            total = total + self.target_weights.get(key, 1.0) * loss_k

        if self.correction_l2_weight > 0.0 and "correction" in predictions:
            correction = predictions["correction"]
            total = total + self.correction_l2_weight * correction.square().mean()

        if self.hole_separation_weight > 0.0:
            total = total + self.hole_separation_weight * self._hole_separation_loss(
                predictions=predictions,
                batch=batch,
                sample_weight=sample_weight,
                device=device,
            )

        if self.sieve_plane_weight > 0.0:
            total = total + self.sieve_plane_weight * self._sieve_plane_loss(
                predictions=predictions,
                batch=batch,
                sample_weight=sample_weight,
                device=device,
            )

        return total

    @torch.no_grad()
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, object],
    ) -> Dict[str, float]:
        """Return simple validation metrics in scaled space."""
        targets = batch["targets"]
        tolerances = batch["tolerances"]
        masks = batch["target_mask"]

        metrics: Dict[str, float] = {}
        for key in _TARGET_KEYS:
            if key not in predictions:
                continue

            pred = predictions[key].squeeze(-1)
            tgt = targets[key].to(pred.device).squeeze(-1)
            tol = tolerances[key].to(pred.device).squeeze(-1).clamp_min(0.0)
            mask = masks[key].to(pred.device).squeeze(-1) > 0.5
            if mask.sum().item() == 0:
                metrics[f"{key}_center_rmse"] = float("nan")
                metrics[f"{key}_deadzone_rmse"] = float("nan")
                metrics[f"{key}_within_tol"] = float("nan")
                continue

            err = pred[mask] - tgt[mask]
            abs_err = err.abs()
            excess = torch.clamp(abs_err - tol[mask], min=0.0)
            metrics[f"{key}_center_rmse"] = float(torch.sqrt(torch.mean(err.square())).item())
            metrics[f"{key}_deadzone_rmse"] = float(torch.sqrt(torch.mean(excess.square())).item())
            metrics[f"{key}_within_tol"] = float((abs_err <= tol[mask]).float().mean().item())

        return metrics
