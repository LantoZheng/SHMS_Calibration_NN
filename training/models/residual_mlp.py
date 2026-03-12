"""
ResidualMLP — Core neural network architecture for SHMS optics calibration.

Architecture:
    Input(input_dim) → n_residual_blocks × ResidualBlock(hidden_dim) →
    4 multi-task output heads [delta, xptar, yptar, ytar]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Single residual block: Linear → SiLU → Linear → add skip → SiLU."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.fc1(x))
        out = self.fc2(out)
        return self.act2(out + residual)


class ResidualMLP(nn.Module):
    """
    Multi-task Residual MLP for SHMS optics reconstruction.

    Parameters
    ----------
    input_dim : int
        Number of input features. Default 6:
        [x_fp, y_fp, xp_fp, yp_fp, x_tar, p0].
    hidden_dim : int
        Width of all hidden layers. Default 256.
    n_residual_blocks : int
        Number of residual blocks in the backbone. Default 4.
    branch_dim : int
        Width of the intermediate layer in each output head. Default 64.

    Outputs
    -------
    forward() returns a dict with keys:
        'delta', 'xptar', 'yptar', 'ytar' — each tensor of shape (batch, 1).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        n_residual_blocks: int = 4,
        branch_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_residual_blocks = n_residual_blocks
        self.branch_dim = branch_dim

        # ── Backbone ────────────────────────────────────────────────────
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.backbone = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)]
        )

        # ── Multi-task output heads ──────────────────────────────────────
        self.head_delta = self._make_head(hidden_dim, branch_dim)
        self.head_xptar = self._make_head(hidden_dim, branch_dim)
        self.head_yptar = self._make_head(hidden_dim, branch_dim)
        self.head_ytar = self._make_head(hidden_dim, branch_dim)

    @staticmethod
    def _make_head(hidden_dim: int, branch_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.SiLU(),
            nn.Linear(branch_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_layer(x)
        for block in self.backbone:
            h = block(h)

        return {
            "delta": self.head_delta(h),
            "xptar": self.head_xptar(h),
            "yptar": self.head_yptar(h),
            "ytar": self.head_ytar(h),
        }

    # ── Freeze / unfreeze helpers ────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze input_layer and all residual blocks; leave heads trainable."""
        for param in self.input_layer.parameters():
            param.requires_grad = False
        for block in self.backbone:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Summary ─────────────────────────────────────────────────────────

    def model_summary(self) -> None:
        """Print parameter counts per component."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        print("ResidualMLP parameter summary")
        print(f"  input_layer       : {count(self.input_layer):>10,}")
        for i, block in enumerate(self.backbone):
            print(f"  residual_block[{i}] : {count(block):>10,}")
        print(f"  head_delta        : {count(self.head_delta):>10,}")
        print(f"  head_xptar        : {count(self.head_xptar):>10,}")
        print(f"  head_yptar        : {count(self.head_yptar):>10,}")
        print(f"  head_ytar         : {count(self.head_ytar):>10,}")
        print(f"  TOTAL             : {count(self):>10,}")
