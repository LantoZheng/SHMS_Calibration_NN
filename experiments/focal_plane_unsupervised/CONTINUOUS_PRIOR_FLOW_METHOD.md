# 基于连续光学弱先验的可逆 FP 5D 聚类度量

## 目标

目标不是取代现有的 sieve-plane 聚类真值，而是回答一个较具体的问题：在只观测到 focal-plane 的五个原始变量时，能否利用当前光学校准产生的**连续**重建量，将事件送入一个仍然是五维、但对 sieve-hole 更平直的空间，再在该空间中聚类。

输入变量为：

$$
x=(xfp,\;yfp,\;xpfp,\;ypfp,\;fr\_ybpm)\in\mathbb{R}^{5}.
$$

输出也是五维：

$$
z=F(x)\in\mathbb{R}^{5}.
$$

因此这不是把 FP 压到二维或三维的 embedding。`F` 是可逆变换，所有五个自由度均被保留；在最终距离中，后两个残差轴也保留非零权重。

## 为什么原始 FP 5D 直接聚类困难

已有 sieve-plane 聚类提供了 220 个 `(foil, hole)` 参考中心。把每个参考 cluster 的事件在 FP 中取平均后，中心点显示出很规则的结构：每个 foil 对应一张弯曲的、约二维的中心流面。

| foil | 中心数 | 前两个中心主方向方差占比 | 中心有效维数 | FP 5D 距离与 sieve 距离的 Spearman 相关 |
|---:|---:|---:|---:|---:|
| 0 | 66 | 0.748 | 2.903 | 0.607 |
| 1 | 77 | 0.814 | 2.685 | 0.589 |
| 2 | 77 | 0.801 | 2.357 | 0.530 |

对每个中心，FP 5D 中的最近另一中心有 65–83% 是 sieve 网格相邻 hole。这说明中心级映射具有局部连续性；但同一格点在不同 foil 的中心距离仅为 robust-scaled 5D 中约 0.33–0.46，三张流面彼此接近。更重要的是，单个 hole 的事件云比其中心流面厚得多，会使三张流面局部重叠、折叠。

这解释了为什么原始 FP 5D 上的 DBSCAN、HDBSCAN、谱图、局部切空间图与 SSC 都不能可靠恢复 hole：它们直接按事件云的厚度和相交区连接，而不是按去噪后的中心流面距离连接。

## `fr_ybpm` 的角色

原始 5D 等权距离还把 `fr_ybpm` 当作与四个光学变量同等的几何坐标。以现有 `(foil, hole)` 参考群组做方差分解，情况并非如此：

| 变量 | 中心间方差占比 | hole 内方差占比 | 中心信号/展宽 |
|---|---:|---:|---:|
| `xfp` | 0.110 | 0.894 | 0.123 |
| `yfp` | 0.405 | 0.604 | 0.671 |
| `xpfp` | 0.340 | 0.663 | 0.512 |
| `ypfp` | 0.853 | 0.231 | 3.689 |
| `fr_ybpm` | 0.003 | 0.997 | 0.003 |

`fr_ybpm` 几乎不移动参考 hole 中心，却显著增加事件云厚度。因此它不应作为与其他变量等权的 hole 分离方向；但它也不应被简单丢弃，因为它携带 raster 条件。这里采用“条件变量而非主距离轴”的处理。

## 训练数据与禁止使用的信息

本实验从 run 25521 数据中随机取 60,000 个事件。按完整的 `(foil_position, cluster)` 参考群组随机留出 20%：

- 训练：47,467 个事件；
- 测试：12,533 个事件；
- 测试包含 44 个训练阶段完全未见的参考 hole。

训练过程**不读取也不使用**下列字段：

- `cluster_center_x`、`cluster_center_y`；
- `foil_ytar_center`；
- `hole_id`、`hole_row`、`hole_col`；
- `cluster`、`foil_position`。

这些字段仅用于构造严格的整 hole 留出划分，以及在训练后评价结果。

训练允许使用的弱监督目标是逐事件、连续的当前光学重建：

$$
y=(sieve_x,\;sieve_y,\;P\_gtr\_y).
$$

它们不是离散 hole 标签，也不是机械格点中心；但确实是当前光学校准的输出。因此本方法的定位是“以连续重建为弱先验，重构相同几何”，不是完全无重建信息的独立发现法。

## 第一步：raster 条件校正，但保持五维

仅在训练集上拟合一个三次样条加 Ridge 回归：

$$
\hat o(fr\_ybpm)\approx(xfp,yfp,xpfp,ypfp).
$$

随后构造：

$$
x'=(o-\hat o(fr\_ybpm),\;fr\_ybpm)\in\mathbb{R}^{5}.
$$

也就是说，前四维是去除平滑 raster 展宽后的光学残差，第五维仍保留原始 `fr_ybpm`。该步骤只用训练事件拟合，避免测试集信息泄漏。

单独使用该校正后的 5D 不足以解决聚类问题：测试集的同参考 hole 20-NN 纯度仍为 0.393，silhouette 为 -0.168。它是稳健化预处理，不是独立聚类器。

## 第二步：连续弱监督的可逆 5D transport

使用六个 affine coupling layer 组成的 RealNVP：

$$
F:\mathbb{R}^{5}\rightarrow\mathbb{R}^{5}.
$$

每个 coupling layer 保留一部分坐标，使用其余坐标的非线性网络来缩放和平移另一个子集；交替 mask 后，整体映射可逆。网络输出定义为：

$$
z=F(x')=(z_1,z_2,z_3,z_4,z_5).
$$

训练损失为：

$$
\mathcal{L}=\operatorname{MSE}((z_1,z_2,z_3),\tilde y)
+0.02\,\operatorname{MSE}((z_4,z_5),(x'_4,x'_5)),
$$

其中 `\tilde y` 是标准化后的连续弱目标。前 3 个输出轴学习与 sieve/target-y 对齐；后 2 个轴保留为残差自由度，避免模型把原始 5D 强行压成 3D。

训练 100 epoch 后，数值 round-trip 最大绝对误差为 `6.84e-05`，验证了映射在数值上保持可逆。

## 最终 5D 距离

最终聚类不只使用前三个预测轴，而是使用：

$$
d^2(i,j)=\sum_{k=1}^{3}(z_{ik}-z_{jk})^2
+0.10\sum_{k=4}^{5}(z_{ik}-z_{jk})^2.
$$

后两轴权重为 0.10，而非零；它们保留可逆残差信息，但不让 raster/非目标自由度重新主导 hole 几何。

## 留出 hole 的结果

| 测试空间 | 连续目标 kNN 重叠 | 同参考 hole 20-NN 纯度 | 参考 hole silhouette |
|---|---:|---:|---:|
| raster 条件校正后的原始 5D | 0.045 | 0.393 | -0.168 |
| flow 后完整 5D、等权 | 0.390 | 0.990 | 0.439 |
| flow 后完整 5D、残差权重 0.10 | **0.432** | **0.991** | **0.506** |

在最后一种完整 5D 距离上运行 HDBSCAN，并扫描 `min_cluster_size={15,30,60}` 与 `eom/leaf` 选择方式。按“至少 20 簇、噪声少于 50%、最大簇少于 10%”筛选后，最佳候选为：

| 指标 | 数值 |
|---|---:|
| HDBSCAN cluster 数 | 47 |
| 参考留出 hole 数 | 44 |
| 噪声率 | 4.62% |
| 中位 cluster 大小 | 241 |
| 最大 cluster 占比 | 5.62% |
| AMI | 0.952 |
| ARI | 0.916 |

这表明：在没有看过测试 hole 的离散身份或中心位置的前提下，连续弱先验加可逆变换已足以把原始事件云展开到与现有 sieve-plane 分组高度一致的完整 5D 聚类空间。

## 与此前方案的关系

此前纯 FP 的图、diffusion、SSC 和局部切空间方法直接在厚事件云上操作，最佳 AMI 约为 0.20–0.24；无监督局部高斯 mixture 可达到约 0.45，但其自然成分对应的是局部流片，而不是 hole。

当前方法利用这些观察结果，但不再要求无监督算法自己猜出“如何展开流面”：连续 `sieve_x/y` 和 `P_gtr_y` 提供展开方向，RealNVP 保留剩余信息，HDBSCAN 只在展开后完成离散分割。

## 适用边界与风险

1. `sieve_x/y` 与 `P_gtr_y` 来自当前光学校准。高 AMI/ARI 证明的是对当前重建几何的可迁移插值，不是独立物理真值。
2. 当前验证在同一个 run 内按完整 hole 留出。它排除了事件级记忆和离散 hole 记忆，但不能替代跨 run 验证。
3. 目前只验证了该 run 的抽样和固定随机划分。模型超参数、残差权重、HDBSCAN 参数仍应在训练 run 内确定，再对未见 run 一次性评估。
4. 若未来目标是完全不使用重建量，则应退回无监督 local mixture，并接受其目前只能恢复局部流片、不能稳定恢复 hole 的事实。

## 下一步：跨 run 验证

下一步实验应保持模型逻辑不变：

1. 选一个或多个 run 完全不进入训练、scaler 拟合、raster 条件回归和 HDBSCAN 调参；
2. 使用其余 run 训练连续弱先验 flow；
3. 将未见 run 送入同一 5D transport；
4. 用训练 run 固定的 HDBSCAN 设置聚类；
5. 只在最后与该 run 的 sieve-plane 参考结果比较 AMI、ARI、噪声率、cluster 数和每 hole 纯度。

若跨 run 仍能保持低噪声、接近正确的 cluster 数并显著优于原始 FP 5D，则该方法可以作为当前 sieve-plane 流程的 FP-space 辅助聚类与质量控制模块。

## 复现

核心脚本为 `run_continuous_prior_flow_metric.py`。它依赖 `numpy`、`pandas`、`scikit-learn` 和 CPU PyTorch；本次使用隔离 `uv` 环境运行：

```powershell
$env:UV_CACHE_DIR = "$env:TEMP\codex-uv-focal-observatory"
Push-Location $env:TEMP
uv run --isolated --with numpy --with pandas --with scikit-learn --with torch python "C:\Users\Lanto\Desktop\AI_ML R-SIDIS\SHMS_Calibration_NN\experiments\focal_plane_unsupervised\run_continuous_prior_flow_metric.py"
Pop-Location
```

输出摘要为 `results/continuous_prior_flow_metric_holeholdout_summary.json`，HDBSCAN 扫描为 `results/continuous_prior_flow_hdbscan_holeholdout_scan.csv`。
