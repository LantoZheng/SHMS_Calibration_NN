"""
ResidualTransportMLP — structured transport + residual correction backbone.

This model promotes the notebook-proven `ResMLP_transport` idea into the
formal training package. Its output is decomposed as:

    y_hat = y_linear + y_correction
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.models.residual_mlp import ResidualBlock


_TARGET_KEYS = ["delta", "xptar", "yptar", "ytar"]


class ResidualTransportMLP(nn.Module):
    """Structured optics model with explicit linear transport path."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        n_residual_blocks: int = 4,
        branch_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_residual_blocks = n_residual_blocks
        self.branch_dim = branch_dim
        self.dropout = dropout

        self.linear_path = nn.Linear(input_dim, len(_TARGET_KEYS))
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.backbone = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_residual_blocks)]
        )
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.SiLU(),
            nn.Linear(branch_dim, len(_TARGET_KEYS)),
        )
        self._zero_initialise_correction_head()

    def _zero_initialise_correction_head(self) -> None:
        last_linear = self.correction_head[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        linear = self.linear_path(x)
        h = self.input_projection(x)
        for block in self.backbone:
            h = block(h)
        correction = self.correction_head(h)
        total = linear + correction

        outputs = {key: total[:, i : i + 1] for i, key in enumerate(_TARGET_KEYS)}
        outputs["linear_output"] = linear
        outputs["correction"] = correction
        return outputs

    def freeze_backbone(self) -> None:
        for module in (self.linear_path, self.input_projection, self.backbone):
            for param in module.parameters():
                param.requires_grad = False

    def freeze_linear_path(self) -> None:
        for param in self.linear_path.parameters():
            param.requires_grad = False

    def unfreeze_linear_path(self) -> None:
        for param in self.linear_path.parameters():
            param.requires_grad = True

    def freeze_correction_branch(self) -> None:
        for module in (self.input_projection, self.backbone, self.correction_head):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_correction_branch(self) -> None:
        for module in (self.input_projection, self.backbone, self.correction_head):
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def initialise_linear_path_least_squares(
        self,
        X_train: torch.Tensor | np.ndarray,
        Y_train: torch.Tensor | np.ndarray,
        ridge: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialise the linear path by ridge-regularised least squares."""
        x_np = X_train.detach().cpu().numpy() if isinstance(X_train, torch.Tensor) else np.asarray(X_train)
        y_np = Y_train.detach().cpu().numpy() if isinstance(Y_train, torch.Tensor) else np.asarray(Y_train)

        x_np = np.asarray(x_np, dtype=np.float64)
        y_np = np.asarray(y_np, dtype=np.float64)
        ones = np.ones((x_np.shape[0], 1), dtype=np.float64)
        design = np.concatenate([x_np, ones], axis=1)
        gram = design.T @ design
        gram += ridge * np.eye(gram.shape[0], dtype=np.float64)
        beta = np.linalg.solve(gram, design.T @ y_np)

        weight = beta[:-1, :].T.astype(np.float32)
        bias = beta[-1, :].astype(np.float32)
        self.linear_path.weight.copy_(torch.from_numpy(weight))
        self.linear_path.bias.copy_(torch.from_numpy(bias))
        return weight, bias

    def model_summary(self) -> None:
        def count(module: nn.Module | Iterable[nn.Module]) -> int:
            if isinstance(module, nn.Module):
                return sum(p.numel() for p in module.parameters())
            return sum(sum(p.numel() for p in m.parameters()) for m in module)

        print("ResidualTransportMLP parameter summary")
        print(f"  linear_path       : {count(self.linear_path):>10,}")
        print(f"  input_projection  : {count(self.input_projection):>10,}")
        for i, block in enumerate(self.backbone):
            print(f"  residual_block[{i}] : {count(block):>10,}")
        print(f"  correction_head   : {count(self.correction_head):>10,}")
        print(f"  TOTAL             : {count(self):>10,}")