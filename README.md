# Calibration_NN

Final retained V3 solution on GUI manually labeled data: `ResidualTransportMLP` with 5D inputs `x_fp, y_fp, xp_fp, yp_fp, fry`. The model has been retrained in the `general` Conda environment and compared with XGBoost using the exact same held-out validation set.

## Contents

- `src/training/`: V3 training, relabeling, dataset, loss, and model implementations.
- `configs/common_holes_v3.yaml`: Final reproducible configuration; fixed held-out holes: 59, 1015, 1035, 1073, 2060.
- `data/stage2_soc_gui_labeled.csv`: GUI-labeled training data.
- `models/common_holes_v3/best_finetune.pth`: Final selected checkpoint (epoch 74).
- `results/`: Training history, fixed splits, and unified comparison results against XGBoost.

## Reproduction

```powershell
python src/training/scripts/run_stage2_transport_fullroot.py `
  --config configs/common_holes_v3.yaml `
  --root-file data/stage2_soc_gui_labeled.csv `
  --device cuda
```

For CPU environments, set `--device` to `cpu` and reduce the number of workers in the configuration.

See [ANALYSIS.md](ANALYSIS.md) for detailed selection rationale, and `results/common_benchmark.{json,csv}` for unified metrics.
