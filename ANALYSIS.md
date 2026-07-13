# NN V3 GUI：最终方案与统一结果

## 选择

保留 `common_holes_v3`，它继承 V3 Baseline 架构和训练日程，并在 `general` 环境中从同一预训练模型重新训练。它与 XGBoost 共用 GUI 数据、5D 特征、target/tolerance 定义和固定的留孔验证划分。

## 架构与训练

模型为 `ResidualTransportMLP(5, 192, 4 residual blocks, branch_dim=64, dropout=0.10)`：显式线性 transport path 加残差 correction branch。训练先仅训练输出头，第 21 epoch 解冻 correction branch；使用 Huber、hole-separation 与 sieve-plane 损失，训练 80 epochs，batch size 2048。

验证固定留出 5 个 hole：59、1015、1035、1073、2060；共 118,655 条训练事件和 2,679 条验证事件。最佳 checkpoint 在第 74 epoch 保存，训练日志、history、split 和配置位于 `results/common_holes_v3/`。

## 统一对比（最终标准）

在 `results/figures/metrics.json` 的整体评估中：

| 方案 | ytar RMSE (cm) | xptar RMSE (rad) | yptar RMSE (rad) | delta RMSE (%) |
| --- | ---: | ---: | ---: | ---: |
| ResidualTransportMLP V3 | **1.54575** | **0.001785** | **0.001117** | **0.26689** |
| XGBoost multi-output tree | 3.28241 | 0.014734 | 0.033086 | 12.32926 |

同一评估还给出 xptar/yptar 容差命中率 60.3%/74.7%，XGBoost 为 15.6%/20.3%。因此 NN 是当前唯一应保留为最终主模型的方案。完整、每 foil 指标在 `results/common_benchmark.json`。
