"""
ONNX export utilities for ResidualMLP.

Because ResidualMLP.forward() returns a dict, we use a thin wrapper that
stacks all four outputs into a single (batch, 4) tensor with the order:
    [delta, xptar, yptar, ytar]

The scaler bundle is saved alongside the ONNX file as
{output_path}.scaler.json so that C++ consumers can perform the same
normalisation/de-normalisation.
"""

from __future__ import annotations

import os

import numpy as np


def export_to_onnx(
    model,
    scaler_bundle,
    output_path: str,
    input_dim: int = 6,
    opset_version: int = 17,
) -> None:
    """
    Export *model* to ONNX format.

    Parameters
    ----------
    model         : trained ResidualMLP (will be set to eval mode).
    scaler_bundle : ScalerBundle; saved alongside as {output_path}.scaler.json.
    output_path   : destination .onnx file path.
    input_dim     : number of input features (must match model.input_dim).
    opset_version : ONNX opset version (default 17).
    """
    import torch
    import torch.nn as nn

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "onnx is required for ONNX export. Install with: pip install onnx"
        ) from exc

    model.eval()

    # Wrapper: dict → stacked tensor ----------------------------------------
    class _StackedWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.inner(x)
            return torch.cat(
                [out["delta"], out["xptar"], out["yptar"], out["ytar"]], dim=1
            )

    wrapper = _StackedWrapper(model)
    wrapper.eval()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dummy_input = torch.zeros(1, input_dim)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["focal_plane_features"],
        output_names=["optics_targets"],
        dynamic_axes={
            "focal_plane_features": {0: "batch_size"},
            "optics_targets": {0: "batch_size"},
        },
    )

    # Save scaler bundle alongside ONNX model --------------------------------
    scaler_path = output_path + ".scaler.json"
    scaler_bundle.save(scaler_path)

    print(f"ONNX model saved to: {output_path}")
    print(f"Scaler bundle saved to: {scaler_path}")
    print("\nONNX I/O names for C++ integration:")
    print("  Input  : 'focal_plane_features'  shape=(batch, {})".format(input_dim))
    print("  Output : 'optics_targets'         shape=(batch, 4)")
    print("  Output column order: [delta, xptar, yptar, ytar]")


def verify_onnx_export(onnx_path: str, test_input: np.ndarray) -> np.ndarray:
    """
    Load the exported ONNX model with onnxruntime and run a test inference.

    Parameters
    ----------
    onnx_path  : path to the .onnx file.
    test_input : numpy array of shape (N, input_dim).

    Returns
    -------
    numpy array of shape (N, 4) with columns [delta, xptar, yptar, ytar].
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Install with: pip install onnxruntime"
        ) from exc

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: test_input.astype(np.float32)})[0]
    return output
