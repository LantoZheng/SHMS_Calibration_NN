# SHMS Optics Two-Stage Training Framework

Production-grade neural network training pipeline for SHMS (Super High Momentum
Spectrometer) optics calibration at Jefferson Lab Hall C.

## Directory Structure

```
training/
├── configs/
│   ├── pretrain_config.yaml      # Stage 1 hyperparameters
│   └── finetune_config.yaml      # Stage 2 hyperparameters
├── data/
│   ├── simc_dataset.py           # SIMC ROOT → PyTorch Dataset
│   ├── sieve_dataset.py          # Labeled sieve/foil data → PyTorch Dataset
│   └── preprocessing.py          # ScalerBundle, feature utilities
├── models/
│   ├── residual_mlp.py           # ResidualMLP architecture
│   └── physics_loss.py           # Physics-informed loss (TRANSPORT matrix)
├── trainers/
│   ├── pretrain.py               # Stage 1 training loop
│   └── finetune.py               # Stage 2 fine-tuning loop
├── evaluation/
│   └── metrics.py                # OpticsEvaluator (metrics + plots)
├── export/
│   └── onnx_export.py            # ONNX export + verification
├── scripts/
│   ├── run_pretrain.py           # CLI: Stage 1
│   ├── run_finetune.py           # CLI: Stage 2
│   └── evaluate_model.py         # CLI: evaluation & comparison
└── tests/
    ├── test_model.py
    ├── test_dataset.py
    └── test_loss.py
```

## Installation

Install the base package first, then the ML extras:

```bash
pip install -e ".[training]"
# or for all extras:
pip install torch>=2.0.0 onnx>=1.14.0 onnxruntime>=1.16.0 pyyaml>=6.0 \
            matplotlib seaborn uproot awkward
```

## Stage 1 — Pre-training on SIMC Data

Pre-trains the `ResidualMLP` backbone on large-statistics SIMC Monte-Carlo
events.  This gives the model a physics-consistent prior before it sees
real calibration data.

```bash
python training/scripts/run_pretrain.py \
    --config training/configs/pretrain_config.yaml \
    --simc-files /path/to/simc_run_*.root \
    --output-dir checkpoints/pretrain/ \
    --p0 4.4 \
    --device cuda
```

### Expected SIMC Variables

| SIMC branch | Physical meaning            | Unit  |
|-------------|-----------------------------|-------|
| `hsxfp`     | Focal-plane x               | cm    |
| `hsyfp`     | Focal-plane y               | cm    |
| `hsxpfp`    | Focal-plane x′ (slope)      | rad   |
| `hsypfp`    | Focal-plane y′ (slope)      | rad   |
| `hsdeltai`  | Thrown δ = (p−p₀)/p₀       | %     |
| `hsxptari`  | Thrown θ at target          | rad   |
| `hsyptari`  | Thrown φ at target          | rad   |
| `hsztari`   | Thrown z_react (≈ y_tar)    | cm    |

`x_tar` is synthesised as Gaussian noise with σ = `x_tar_sigma_cm` (raster model).
`p0` is appended as a constant column.

The fitted `ScalerBundle` is saved to `checkpoints/pretrain/scaler_bundle.json`
and **must be reused** in Stage 2.

## Stage 2 — Fine-tuning on Sieve/Foil Data

Fine-tunes only the output heads (backbone frozen) on labeled data from
`SHMS_Optics_calibration_tools`.

```bash
python training/scripts/run_finetune.py \
    --config training/configs/finetune_config.yaml \
    --sieve-data /path/to/labeled_sieve_data.csv \
    --pretrained-checkpoint checkpoints/pretrain/best_pretrain.pth \
    --p0 4.4 \
    --device cuda
```

### Expected Input Format (hcana column names)

| Column       | Physical meaning            | Unit  |
|--------------|-----------------------------|-------|
| `P_dc_x_fp`  | Focal-plane x               | cm    |
| `P_dc_y_fp`  | Focal-plane y               | cm    |
| `P_dc_xp_fp` | Focal-plane x′              | rad   |
| `P_dc_yp_fp` | Focal-plane y′              | rad   |
| `P_react_x`  | Target-plane x (optional)   | cm    |
| `P_gtr_dp`   | Reconstructed δ             | %     |
| `P_gtr_th`   | Reconstructed θ at target   | rad   |
| `P_gtr_ph`   | Reconstructed φ at target   | rad   |
| `P_react_z`  | Reaction vertex z           | cm    |

This CSV is the output of `SHMS_Optics_calibration_tools` cluster labeling.

### Variable Naming Convention

| SIMC name   | hcana name      | Internal key | Physical meaning             |
|-------------|-----------------|--------------|------------------------------|
| `hsxfp`     | `P_dc_x_fp`     | `x_fp`       | Focal-plane x (cm)           |
| `hsyfp`     | `P_dc_y_fp`     | `y_fp`       | Focal-plane y (cm)           |
| `hsxpfp`    | `P_dc_xp_fp`    | `xp_fp`      | Focal-plane x′ (rad)         |
| `hsypfp`    | `P_dc_yp_fp`    | `yp_fp`      | Focal-plane y′ (rad)         |
| —           | `P_react_x`     | `x_tar`      | Target x (cm)                |
| —           | —               | `p0`         | Central momentum (GeV/c)     |
| `hsdeltai`  | `P_gtr_dp`      | `delta`      | δ = (p−p₀)/p₀ (%)           |
| `hsxptari`  | `P_gtr_th`      | `xptar`      | In-plane angle θ (rad)       |
| `hsyptari`  | `P_gtr_ph`      | `yptar`      | Out-of-plane angle φ (rad)   |
| `hsztari`   | `P_react_z`     | `ytar`       | Reaction vertex z (cm)       |

## Evaluation

Compare the trained NN against the polynomial baseline:

```bash
python training/scripts/evaluate_model.py \
    --checkpoint checkpoints/finetune/best_finetune.pth \
    --test-data /path/to/test_data.csv \
    --poly-coeffs models/poly_coeffs_20260112_202706.json \
    --scaler-bundle checkpoints/pretrain/scaler_bundle.json \
    --plot-dir plots/
```

Output: per-target RMSE / R² comparison table, residual histograms with
Gaussian fits, and sieve-hole reconstruction scatter plot.

## ONNX Export

```python
from training.models.residual_mlp import ResidualMLP
from training.data.preprocessing import ScalerBundle
from training.export.onnx_export import export_to_onnx, verify_onnx_export
import torch, numpy as np

# Load trained model
ckpt = torch.load("checkpoints/finetune/best_finetune.pth")
model = ResidualMLP(**ckpt["config"]["model"])
model.load_state_dict(ckpt["model_state_dict"])

scaler = ScalerBundle.load("checkpoints/pretrain/scaler_bundle.json")
export_to_onnx(model, scaler, "export/shms_optics.onnx", input_dim=6)

# Verify
test_in = np.random.randn(4, 6).astype(np.float32)
out = verify_onnx_export("export/shms_optics.onnx", test_in)
print(out.shape)  # (4, 4)  — [delta, xptar, yptar, ytar]
```

**C++ integration** — Use the ONNX Runtime C++ API with:
- Input name: `"focal_plane_features"`, shape `(batch, 6)`
- Output name: `"optics_targets"`, shape `(batch, 4)`
- Apply scaler normalisation (from `scaler_bundle.json`) before inference
  and inverse-normalisation after.

## Running Tests

```bash
pip install pytest
pytest training/tests/ -v
```

## Architecture Summary

```
Input (batch, 6)
    │
    ▼
Linear(6 → 256)
    │
    ▼  ×4
ResidualBlock:  Linear(256→256) → SiLU → Linear(256→256) → +skip → SiLU
    │
    ├──▶ head_delta  : Linear(256→64) → SiLU → Linear(64→1)
    ├──▶ head_xptar  : Linear(256→64) → SiLU → Linear(64→1)
    ├──▶ head_yptar  : Linear(256→64) → SiLU → Linear(64→1)
    └──▶ head_ytar   : Linear(256→64) → SiLU → Linear(64→1)
```

**Total parameters** (default config): ~560 k

**Physics-informed loss**: Huber data loss + λ × TRANSPORT-matrix Jacobian penalty.
The penalty constrains the model's ∂(output)/∂(input) Jacobian to match
known first-order optics matrix elements.
