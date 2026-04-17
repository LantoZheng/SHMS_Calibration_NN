"""Model exports and factory helpers for SHMS training."""

from __future__ import annotations

from typing import Any, Dict, Optional

from training.models.residual_mlp import ResidualMLP
from training.models.residual_transport_mlp import ResidualTransportMLP


def build_model_from_config(model_config: Optional[Dict[str, Any]], input_dim: int):
	"""Instantiate the requested backbone from a config dictionary."""
	cfg = dict(model_config or {})
	name = str(
		cfg.get("name")
		or cfg.get("backbone")
		or cfg.get("type")
		or "residual_mlp"
	).lower()

	common = {
		"input_dim": input_dim,
		"hidden_dim": cfg.get("hidden_dim", 256),
		"n_residual_blocks": cfg.get("n_residual_blocks", 4),
		"branch_dim": cfg.get("branch_dim", 64),
		"dropout": cfg.get("dropout", 0.0),
	}

	if name in {"residual_transport_mlp", "resmlp_transport", "transport"}:
		return ResidualTransportMLP(**common)
	if name in {"residual_mlp", "resmlp", "baseline"}:
		return ResidualMLP(**common)
	raise ValueError(f"Unsupported model backbone: {name}")


__all__ = [
	"ResidualMLP",
	"ResidualTransportMLP",
	"build_model_from_config",
]
