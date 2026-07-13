# Calibration_NN

最终保留更新后 GUI 手动标注数据上的 V3 方案：`ResidualTransportMLP` 与 5D 输入 `x_fp, y_fp, xp_fp, yp_fp, fry`。该模型已在 `general` Conda 环境中重新训练，并与 XGBoost 使用完全相同的留孔验证集比较。

## 内容

- `src/training/`：V3 训练、重标注、数据集、损失和模型实现。
- `configs/common_holes_v3.yaml`：最终可复现配置；固定保留 hole 59、1015、1035、1073、2060。
- `data/stage2_soc_gui_labeled.csv`：GUI 标注训练数据。
- `models/common_holes_v3/best_finetune.pth`：最终选定 checkpoint（第 74 epoch）。
- `results/`：训练历史、固定划分和与 XGBoost 的统一对比结果。

## 复现

```powershell
python src/training/scripts/run_stage2_transport_fullroot.py `
  --config configs/common_holes_v3.yaml `
  --root-file data/stage2_soc_gui_labeled.csv `
  --device cuda
```

CPU 环境可将 `--device` 改为 `cpu`，并降低配置中的 worker 数量。

详细选择依据见 [ANALYSIS.md](ANALYSIS.md)，统一指标见 `results/common_benchmark.{json,csv}`。
