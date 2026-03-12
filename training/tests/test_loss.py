"""Unit tests for PhysicsInformedLoss."""

import pytest
import torch

from training.models.physics_loss import PhysicsInformedLoss
from training.models.residual_mlp import ResidualMLP


_TARGETS = ["delta", "xptar", "yptar", "ytar"]


def _make_preds_targets(batch: int = 16):
    preds = {k: torch.randn(batch, 1) for k in _TARGETS}
    tgts = {k: torch.randn(batch, 1) for k in _TARGETS}
    return preds, tgts


def test_data_loss_huber():
    loss_fn = PhysicsInformedLoss(lambda_physics=0.0, use_huber=True)
    preds, tgts = _make_preds_targets()
    loss = loss_fn(preds, tgts)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0.0


def test_data_loss_mse():
    loss_fn = PhysicsInformedLoss(lambda_physics=0.0, use_huber=False)
    preds, tgts = _make_preds_targets()
    loss = loss_fn(preds, tgts)
    assert loss.item() >= 0.0


def test_loss_is_zero_for_perfect_preds():
    loss_fn = PhysicsInformedLoss(lambda_physics=0.0, use_huber=False)
    preds = {k: torch.zeros(8, 1) for k in _TARGETS}
    tgts = {k: torch.zeros(8, 1) for k in _TARGETS}
    loss = loss_fn(preds, tgts)
    assert loss.item() < 1e-8


def test_per_target_loss_keys():
    loss_fn = PhysicsInformedLoss(lambda_physics=0.0)
    preds, tgts = _make_preds_targets()
    per_target = loss_fn.compute_per_target_loss(preds, tgts)
    assert set(per_target.keys()) == set(_TARGETS)
    for v in per_target.values():
        assert v.item() >= 0.0


def test_physics_penalty_increases_loss():
    transport = {"M_xptar_xp_fp": 0.0, "M_yptar_yp_fp": -1.0}
    loss_pure = PhysicsInformedLoss(lambda_physics=0.0, transport_matrix=None)
    loss_phys = PhysicsInformedLoss(lambda_physics=1.0, transport_matrix=transport)

    model = ResidualMLP(input_dim=6, hidden_dim=32, n_residual_blocks=1, branch_dim=8)
    x = torch.randn(8, 6, requires_grad=True)
    preds = model(x)
    tgts = {k: torch.randn(8, 1) for k in _TARGETS}

    l_pure = loss_pure(preds, tgts, inputs=x)
    l_phys = loss_phys(preds, tgts, inputs=x)

    # Physics penalty adds a non-negative term, so l_phys >= l_pure - epsilon
    assert l_phys.item() >= l_pure.item() - 1e-5


def test_gradient_flows_through_physics_penalty():
    transport = {"M_xptar_xp_fp": 0.0, "M_yptar_yp_fp": -1.0}
    loss_fn = PhysicsInformedLoss(lambda_physics=0.1, transport_matrix=transport)

    model = ResidualMLP(input_dim=6, hidden_dim=32, n_residual_blocks=1, branch_dim=8)
    x = torch.randn(4, 6, requires_grad=True)
    preds = model(x)
    tgts = {k: torch.randn(4, 1) for k in _TARGETS}

    loss = loss_fn(preds, tgts, inputs=x)
    loss.backward()

    # Model parameters should have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_no_physics_penalty_without_transport_matrix():
    loss_fn = PhysicsInformedLoss(lambda_physics=100.0, transport_matrix=None)
    preds, tgts = _make_preds_targets(8)
    loss_no_phys = loss_fn(preds, tgts, inputs=None)

    loss_fn_ref = PhysicsInformedLoss(lambda_physics=0.0, transport_matrix=None)
    loss_ref = loss_fn_ref(preds, tgts, inputs=None)

    # Without transport_matrix, lambda doesn't matter
    assert abs(loss_no_phys.item() - loss_ref.item()) < 1e-6
