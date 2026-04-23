"""
Weak-label losses for stage-2 SHMS fine-tuning.

The full-root stage-2 workflow supervises the model with geometric weak labels
instead of exact point truth:

- sieve-hole labels provide centre + tolerance for xptar / yptar
- foil labels provide centre + tolerance for ytar
- delta can optionally remain a point target with zero tolerance

The loss used here is a dead-zone Huber loss:

    excess = max(|prediction - centre| - tolerance, 0)
    loss   = Huber(excess, 0)

Inside the known physical tolerance window, the sample contributes zero loss.
Outside the window, the penalty grows smoothly and remains robust to outliers.
"""

from __future__ import annotations

from typing import Dict, Optional

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
    ) -> None:
        super().__init__()
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)
        self.target_weights = {
            key: float((target_weights or {}).get(key, 1.0))
            for key in _TARGET_KEYS
        }
        self.correction_l2_weight = float(correction_l2_weight)

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

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, object]) -> torch.Tensor:
        targets = batch["targets"]
        tolerances = batch["tolerances"]
        masks = batch["target_mask"]
        sample_weight = batch.get("weight", None)

        device = next(iter(predictions.values())).device
        total = torch.tensor(0.0, device=device)

        for key in _TARGET_KEYS:
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
