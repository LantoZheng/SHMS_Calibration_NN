from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MLP_METRICS = ROOT_DIR.parent / 'mlp_reduced_data_test' / 'outputs_notebook' / 'notebook_metrics_all_20260409_033420.json'
RESMLP_ROOT_METRICS = ROOT_DIR / 'outputs_notebook' / 'resmlp_metrics_all_20260409_040145.json'
RESMLP_TRANSPORT_METRICS = ROOT_DIR / 'outputs_transport_skip' / 'transport_resmlp_metrics_all_20260409_041954.json'

OUT_ABS = BASE_DIR / 'model_comparison_four_metrics.png'
OUT_IMPROVEMENT = BASE_DIR / 'model_comparison_improvement_vs_mlp.png'

MODEL_ORDER = [
    ('MLP', MLP_METRICS),
    ('ROOT reco', RESMLP_ROOT_METRICS),
    ('ResMLP_root', RESMLP_ROOT_METRICS),
    ('ResMLP_transport', RESMLP_TRANSPORT_METRICS),
]

COLORS = {
    'MLP': '#7f8c8d',
    'ROOT reco': '#111111',
    'ResMLP_root': '#2c7fb8',
    'ResMLP_transport': '#d95f0e',
}

METRIC_SPECS = [
    ('rmse', 'RMSE', 1.0),
    ('mae', 'MAE', 1.0),
    ('rel_rmse', 'Relative RMSE (%)', 100.0),
    ('rel_mae', 'Relative MAE (%)', 100.0),
]


def load_payload(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def get_metric_values(model_name: str, metric_key: str) -> np.ndarray:
    """
    ROOT reco 的数值取自 ResMLP_root notebook 保存的 baseline_* 指标。
    该 notebook 在载入 ROOT 分支时已对 ztar/ytar 对应分支做过符号对齐
    （即使用了取反后的 reco 量，例如 -psztar；若数据前缀为 pf，同样应对 pfztar 取反）。
    """
    payload = payloads['ResMLP_root'] if model_name == 'ROOT reco' else payloads[model_name]

    if model_name == 'ROOT reco':
        metric_alias = {
            'rmse': 'baseline_rmse',
            'mae': 'baseline_mae',
            'rel_rmse': 'baseline_rel_rmse',
            'rel_mae': 'baseline_rel_mae',
        }
        return np.array(payload['metrics'][metric_alias[metric_key]], dtype=float)

    return np.array(payload['metrics'][metric_key], dtype=float)


payloads = {name: load_payload(path) for name, path in MODEL_ORDER}
targets = payloads['MLP']['target_names']

plt.style.use('seaborn-v0_8-whitegrid')

# ----- 图 1：绝对性能对比 -----
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.ravel()
x = np.arange(len(targets))
width = 0.20

offset_center = (len(MODEL_ORDER) - 1) / 2.0

for ax, (metric_key, metric_label, scale) in zip(axes, METRIC_SPECS):
    for idx, (model_name, _) in enumerate(MODEL_ORDER):
        values = get_metric_values(model_name, metric_key) * scale
        ax.bar(
            x + (idx - offset_center) * width,
            values,
            width=width,
            label=model_name,
            color=COLORS[model_name],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_title(metric_label)
    ax.set_ylabel(metric_label)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
fig.suptitle('MLP vs ROOT reco vs ResMLP_root vs ResMLP_transport\nPerformance Comparison Across Four Targets', fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_ABS, dpi=180, bbox_inches='tight')
plt.close(fig)

# ----- 图 2：相对 MLP 改善百分比 -----
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.ravel()
comp_models = ['ROOT reco', 'ResMLP_root', 'ResMLP_transport']
width = 0.24
mlp_payload = payloads['MLP']

offset_center = (len(comp_models) - 1) / 2.0

for ax, (metric_key, metric_label, scale) in zip(axes, METRIC_SPECS):
    baseline = np.array(mlp_payload['metrics'][metric_key], dtype=float) * scale
    for idx, model_name in enumerate(comp_models):
        values = get_metric_values(model_name, metric_key) * scale
        improvement = 100.0 * (baseline - values) / np.maximum(np.abs(baseline), 1e-12)
        ax.bar(
            x + (idx - offset_center) * width,
            improvement,
            width=width,
            label=f'{model_name} vs MLP',
            color=COLORS[model_name],
            alpha=0.9,
        )

    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_title(f'{metric_label} improvement vs MLP')
    ax.set_ylabel('Improvement (%)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
fig.suptitle('Relative Improvement Over MLP\nPositive = Better Than MLP', fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_IMPROVEMENT, dpi=180, bbox_inches='tight')
plt.close(fig)

print('Saved comparison figures:')
print('-', OUT_ABS)
print('-', OUT_IMPROVEMENT)
