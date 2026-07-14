# NN V3 GUI: Final Solution and Unified Results

## Selection

Keep `common_holes_v3`, which inherits the V3 Baseline architecture and training schedule, and is retrained from the same pretrained model in the `general` environment. It shares GUI data, 5D features, target/tolerance definitions, and a fixed hold-out validation split with XGBoost.

## Architecture & Training

The model is `ResidualTransportMLP(5, 192, 4 residual blocks, branch_dim=64, dropout=0.10)`: an explicit linear transport path plus a residual correction branch. Training first trains only the output head, then unfreezes the correction branch at epoch 21; uses Huber, hole-separation, and sieve-plane losses, trained for 80 epochs with a batch size of 2048.

Validation uses a fixed hold-out of 5 holes: 59, 1015, 1035, 1073, 2060; totaling 118,655 training events and 2,679 validation events. The best checkpoint is saved at epoch 74. Training logs, history, split, and configuration are located in `results/common_holes_v3/`.

## Unified Comparison (Final Standard)

Overall evaluation from `results/figures/metrics.json`:

| Solution | ytar RMSE (cm) | xptar RMSE (rad) | yptar RMSE (rad) | delta RMSE (%) |
| --- | ---: | ---: | ---: | ---: |
| ResidualTransportMLP V3 | **1.54575** | **0.001785** | **0.001117** | **0.26689** |
| XGBoost multi-output tree | 3.28241 | 0.014734 | 0.033086 | 12.32926 |

The same evaluation also reports xptar/yptar tolerance hit rates of 60.3%/74.7%, compared to XGBoost's 15.6%/20.3%. Therefore, the NN is the only solution that should be retained as the final primary model. Full per-foil metrics are available in `results/common_benchmark.json`.
