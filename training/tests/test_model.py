"""Unit tests for ResidualMLP architecture."""

import pytest
import torch

from training.models.residual_mlp import ResidualMLP


@pytest.fixture
def model():
    return ResidualMLP(input_dim=6, hidden_dim=64, n_residual_blocks=2, branch_dim=16)


def test_forward_output_keys(model):
    x = torch.randn(8, 6)
    out = model(x)
    assert set(out.keys()) == {"delta", "xptar", "yptar", "ytar"}


def test_forward_output_shapes(model):
    batch = 16
    x = torch.randn(batch, 6)
    out = model(x)
    for key, tensor in out.items():
        assert tensor.shape == (batch, 1), f"{key}: expected ({batch}, 1), got {tensor.shape}"


def test_forward_single_event(model):
    x = torch.randn(1, 6)
    out = model(x)
    for tensor in out.values():
        assert tensor.shape == (1, 1)


def test_custom_input_dim():
    m = ResidualMLP(input_dim=4, hidden_dim=32, n_residual_blocks=1, branch_dim=8)
    x = torch.randn(5, 4)
    out = m(x)
    assert all(v.shape == (5, 1) for v in out.values())


def test_freeze_backbone(model):
    model.freeze_backbone()
    # Input layer and blocks should be frozen
    for param in model.input_layer.parameters():
        assert not param.requires_grad, "input_layer should be frozen"
    for block in model.backbone:
        for param in block.parameters():
            assert not param.requires_grad, "backbone block should be frozen"
    # Output heads should still be trainable
    for head in [model.head_delta, model.head_xptar, model.head_yptar, model.head_ytar]:
        for param in head.parameters():
            assert param.requires_grad, "output head should be trainable"


def test_unfreeze_all(model):
    model.freeze_backbone()
    model.unfreeze_all()
    for param in model.parameters():
        assert param.requires_grad, "all params should be trainable after unfreeze_all()"


def test_model_summary_runs(model, capsys):
    model.model_summary()
    captured = capsys.readouterr()
    assert "TOTAL" in captured.out


def test_gradient_flows(model):
    x = torch.randn(4, 6, requires_grad=True)
    out = model(x)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
