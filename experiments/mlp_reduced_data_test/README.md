# MLP Calibration Sanity Test (Corrected Baseline)

这个目录用于快速验证：在 **单个 SIMC ROOT 文件** 上，MLP 是否能学习 SHMS optics 映射（采用修正后的物理一致流程）。

## 目标

- 使用 `h10` 树中的焦平面变量，学习目标面重建量。
- 支持自动识别分支前缀（`ps*` 或 `hs*`）。
- 默认只保留 `stop_id == 0` 的成功传输事件。
- 默认训练 `core3` 目标：`delta, xptar, yptar`（避免 `ytar` 在当前特征下不可学导致塌缩）。
- 支持自动从 `.inp` 解析 `p0`（MeV/c → GeV/c）。
- 默认 `x_tar=0`（避免人为随机噪声）。

## 输入/输出定义

- 输入特征（6维）：`x_fp, y_fp, xp_fp, yp_fp, x_tar, p0`
- 输出目标（默认3维）：`delta, xptar, yptar`（`--target-mode core3`）
- 也支持：`all`（4维）或 `delta-only`（1维）
- `x_tar` 默认常数 0；可切换 `--x-tar-mode random`

## 运行示例

在 `SHMS_Calibration_NN` 根目录执行：

- 推荐基线（自动解析 `p0` + 过滤 `stop_id` + `core3`）

`python experiments/mlp_reduced_data_test/train_reduced_mlp.py --root-file ../mc-single-arm/worksim/shms_nn_train_3gev15deg.root --inp-file ../mc-single-arm/infiles/shms_nn_train_3gev15deg.inp --target-mode core3 --epochs 40 --max-events 100000`

- 只回归 delta

`python experiments/mlp_reduced_data_test/train_reduced_mlp.py --root-file ../mc-single-arm/worksim/shms_nn_train_3gev15deg.root --inp-file ../mc-single-arm/infiles/shms_nn_train_3gev15deg.inp --target-mode delta-only --epochs 40 --max-events 100000`

- 如需复现实验中的“不过滤失败事件”对照

`python experiments/mlp_reduced_data_test/train_reduced_mlp.py --root-file ../mc-single-arm/worksim/shms_nn_train_3gev15deg.root --target-mode core3 --no-filter-stop-id`

## 输出

结果默认写到：`experiments/mlp_reduced_data_test/outputs/`

- `metrics_*.json`：本次训练配置与验证集 RMSE/MAE
- `rmse_mae_*.png`：各目标 RMSE/MAE 柱状图
- `loss_*.png`：训练/验证损失曲线
- `run_summary_*.txt`：本次运行摘要

## 判读建议

- 若 `core3` 下 parity 紧贴对角线、残差以 0 为中心且窄，说明主映射可学。
- 若切换 `all` 后指标明显变差，通常是 `ytar` 信息不足，不建议在当前特征集强行联合训练。
- 若不过滤 `stop_id` 指标显著恶化，说明数据质量是首要瓶颈。
