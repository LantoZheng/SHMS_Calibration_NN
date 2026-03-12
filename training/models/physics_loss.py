"""
Physics-informed loss function for SHMS optics calibration.

Loss = Data Loss (Huber or MSE per target) + λ × Physics Penalty

The physics penalty enforces known first-order TRANSPORT matrix constraints
by penalising the squared difference between the model's computed Jacobian
elements and their expected theoretical values.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    Combined data loss + TRANSPORT-matrix physics penalty.

    Parameters
    ----------
    lambda_physics : float
        Weight of the physics penalty term. Set to 0 to use pure data loss.
    use_huber : bool
        Use Huber loss (robust to outliers). If False, use MSE.
    huber_delta : float
        Delta parameter for Huber loss.
    transport_matrix : dict or None
        Expected first-order TRANSPORT matrix elements. Supported keys:
            'M_xptar_xp_fp'  — ∂(xptar)/∂(xp_fp)
            'M_yptar_yp_fp'  — ∂(yptar)/∂(yp_fp)
            'M_delta_xp_fp'  — ∂(delta)/∂(xp_fp)
        If None, the physics penalty is skipped.
    """

    # Maps transport_matrix key → (output_key, input_feature_index)
    # Input feature order: [x_fp, y_fp, xp_fp, yp_fp, x_tar, p0]
    _TRANSPORT_MAP: Dict[str, tuple] = {
        "M_xptar_xp_fp": ("xptar", 2),
        "M_yptar_yp_fp": ("yptar", 3),
        "M_delta_xp_fp": ("delta", 2),
    }

    _TARGET_KEYS: List[str] = ["delta", "xptar", "yptar", "ytar"]

    def __init__(
        self,
        lambda_physics: float = 0.01,
        use_huber: bool = True,
        huber_delta: float = 1.0,
        transport_matrix: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.lambda_physics = lambda_physics
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.transport_matrix = transport_matrix or {}

    # ── Internal helpers ─────────────────────────────────────────────────

    def _data_loss_per_target(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_huber:
            return F.huber_loss(pred, tgt, delta=self.huber_delta, reduction="mean")
        return F.mse_loss(pred, tgt)

    def _physics_penalty(
        self,
        predictions: Dict[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute squared Jacobian penalty against expected TRANSPORT elements."""
        if not self.transport_matrix:
            return torch.tensor(0.0, device=inputs.device)

        penalty = torch.tensor(0.0, device=inputs.device)
        for key, expected_val in self.transport_matrix.items():
            if key not in self._TRANSPORT_MAP:
                continue
            out_key, in_idx = self._TRANSPORT_MAP[key]
            output = predictions[out_key]  # (batch, 1)

            # Compute ∂output / ∂inputs[:, in_idx] via autograd
            grad_outputs = torch.ones_like(output)
            (grad,) = torch.autograd.grad(
                outputs=output,
                inputs=inputs,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            if grad is None:
                continue
            computed = grad[:, in_idx].mean()
            expected = torch.tensor(
                float(expected_val), device=inputs.device, dtype=inputs.dtype
            )
            penalty = penalty + (computed - expected) ** 2

        return penalty

    # ── Public API ───────────────────────────────────────────────────────

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : dict  {'delta', 'xptar', 'yptar', 'ytar'} → (batch, 1)
        targets     : dict  same keys → (batch, 1)
        inputs      : tensor (batch, input_dim) with requires_grad=True,
                      required when transport_matrix is non-empty.

        Returns
        -------
        Scalar total loss.
        """
        data_loss = sum(
            self._data_loss_per_target(predictions[k], targets[k])
            for k in self._TARGET_KEYS
            if k in predictions and k in targets
        )

        physics_pen = torch.tensor(0.0, device=data_loss.device)
        if self.transport_matrix and inputs is not None:
            physics_pen = self._physics_penalty(predictions, inputs)

        return data_loss + self.lambda_physics * physics_pen

    def compute_per_target_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return per-target losses (for logging), no physics penalty."""
        return {
            k: self._data_loss_per_target(predictions[k], targets[k])
            for k in self._TARGET_KEYS
            if k in predictions and k in targets
        }
