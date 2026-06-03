# Stage-2 劣化因素归因（2026-06-01）

基于 `tmp_stage2_effects_20260601/evaluate_stage2_effects.py` 的扩展扫描，对用户列出的 5 类改动做如下归因。

## 结论摘要

### 最明确的劣化主因

1. **训练流程里去掉 `ytar`（noytar）**
   - 对照：`nearest_sep_only -> nearest_sep_noytar`
   - `foil1 NN RMSE`: `2.657 -> 4.538 cm`（`+1.881 cm`）
   - `foil1 sigma68`: `2.073 -> 4.450 cm`
   - 结论：**强烈劣化**。

2. **不再保留 holdout foil，改成 all-foils/random split**
   - 对照：`nearest_sep_noytar_delay2 -> nearest_sep_noytar_delay2_allfoils`
   - `foil1 NN RMSE`: `3.521 -> 4.251 cm`（`+0.730 cm`）
   - `foil1 sigma68`: `2.876 -> 3.697 cm`
   - 结论：**强烈劣化**。
   - 说明：虽然 `x/y deadzone` 代理指标继续下降，但 holdout-style foil 表现明显更差，说明选模目标漂移了。

### 明显会带来负面影响，但当前无法与第 4 项完全拆开

3. **从 cluster-center weak labels 切到机械孔/硬 foil center 标签包**
   - 对照：`weaklabel_tolerant_base -> nearest_base`
   - 改动包含：
     - hole 标签从 cluster-center weak labels 变为机械孔中心/tolerance
     - `ytar` 从有 tolerance 的 weak center 变成硬 foil center（0 tolerance）
   - `foil1 NN RMSE`: `2.264 -> 2.687 cm`（`+0.423 cm`）
   - `x/y deadzone mean`: `0.040816 -> 0.048267`（变差）
   - 结论：**这组改动整体带来明显劣化**。

### 不是主因，甚至有轻微帮助的改动

4. **修改 loss（至少当前加入的 `sep` / `sieve-plane loss`）不是劣化主因**
   - `nearest_base -> nearest_sep_only`：`foil1 RMSE` 仅 `-0.029 cm`，影响很小
   - `directgrid_fullnoytar -> directgrid_sieveloss`：`foil1 RMSE` `-0.123 cm`，有中等改善
   - 结论：**loss 修改本身不是导致后来结果普遍变坏的头号原因**。

### 第 4 项（`ytar` 0 容差）

5. **`ytar` 改为 0 容差大概率参与了劣化，但仓库里缺少“只改这一项”的干净 A/B 实验**
   - 现有最接近的对照是：`weaklabel_tolerant_base -> nearest_base`
   - 但这个对照同时还改了 hole 标签定义，因此无法把“硬 `ytar`”单独完全剥离出来。
   - 不过从该对照可见：
     - 切到机械标签 + 硬 `ytar` 后，`foil1` 与 `x/y` 代理指标都一起变差；
     - 而 `check_ytar_vs_foil_z.py` 已确认硬 foil z 本身定义是对的，问题不是 foil z 取值错，而是**把 `ytar` 从宽容弱标签变成 0 容差硬约束后，训练更难、更脆弱**。

## 对 5 类改动的当前排序（按“导致劣化”的嫌疑强弱）

1. **第 3 项：训练流程修改（尤其 `noytar`）** — 最高嫌疑，已被单独实证
2. **第 5 项：不用 holdout foil、改为 all-foils/random split** — 高嫌疑，已被单独实证
3. **第 1+4 项组合：机械标签 + `ytar` 0 容差** — 明显有负面影响，但两者尚未完全拆开
4. **第 2 项：loss 修改** — 不是主因，当前证据显示并不导致主要劣化

## 当前最稳妥的解释

后期结果之所以普遍不如较早方案，主要不是因为 loss 改坏了，而是因为：

- **训练目标被改了**（`noytar`）
- **验证/选模目标被改了**（不再 holdout foil）
- **标签体系整体变硬了**（机械孔中心 + 硬 foil z，且 `ytar` 0 tolerance）

其中前两项的负作用最明确；第三项也有负作用，但还需要更干净的 A/B 才能拆出“机械孔中心标签”和“`ytar` 0 容差”各自占多大比例。
