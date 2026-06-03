# Stage-2 临时影响测试（2026-06-01）

这个临时目录用于回答两个问题：

1. 在 **不改动真实机械孔位**、并且把 **foil 的 z 位置固定为硬标准** 的前提下，哪些后续改动真的影响了结果？
2. 当前 `ytar` 标签/参考量是否真的对应 foil 的实际 z 位置？

## 约束

- **机械 hole position 不改动**：这里只分析既有 checkpoint 与既有标签版本，不在这里生成新的孔位设计表。
- **foil z 位置固定为硬标准**：统一按 `foil0=+10 cm`, `foil1=0 cm`, `foil2=-10 cm` 检查。
- **只做最小消融**：优先复用已有 checkpoint 和已有 `outputs/stage2_ytar_analysis_*`，只补跑缺失的对照项。

## 文件说明

- `check_ytar_vs_foil_z.py`
  - 检查各版 Stage-2 标注中，硬 foil center 是否一致；
  - 比较 `P_react_z` 与 `-P_react_z` 哪个更贴近硬 foil center；
  - 输出 `results/ytar_foil_z_check.json/csv/png`。

- `evaluate_stage2_effects.py`
  - 汇总一组代表性 checkpoint 的 `foil1` / overall `ytar` 表现；
  - 对比 `separation`、`noytar`、`delay2`、`directgrid+max80+eqholeweight`、`sieve-plane loss` 的实际影响；
  - 若缺失某个 `ytar_distribution_summary.json`，会自动调用现有分析脚本补跑；
  - 输出 `results/stage2_effects_summary.csv/json` 与 `results/stage2_effect_comparisons.csv/json/png`。

## 预期重点

- `leave_one_foil_out` vs `random split`
- `nearest_hole_match` vs `directgrid + max80 + eqholeweight`
- `sep` 是否本身就伤害 `foil1`
- `noytar` / `delay2` 是否进一步放大了伤害
- `sieve_plane_loss` 是否真的带来净收益
