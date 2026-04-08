import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

base = Path('/Users/zhengxiaoyang/Desktop/AI_ML R-SIDIS/SHMS_Calibration_NN/experiments/mlp_reduced_data_test')
out_dir = base / 'logbook' / '2026-04-02'
out_dir.mkdir(parents=True, exist_ok=True)

j_all = base / 'experiments/mlp_reduced_data_test/outputs_notebook/notebook_metrics_all_20260402_011516.json'
j_core3 = base / 'experiments/mlp_reduced_data_test/outputs_notebook/notebook_metrics_core3_20260402_012623.json'

with open(j_all) as f:
    all_data = json.load(f)
with open(j_core3) as f:
    core3_data = json.load(f)

labels = ['delta', 'xptar', 'yptar']
rmse_before = np.array(all_data['metrics']['rmse'][:3], dtype=float)
rmse_after = np.array(core3_data['metrics']['rmse'], dtype=float)

x = np.arange(len(labels))
width = 0.34

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(x - width / 2, rmse_before, width, label='Before fix (all, unfiltered)', color='#d62728', alpha=0.85)
ax.bar(x + width / 2, rmse_after, width, label='After fix (core3, stop_id==0)', color='#2ca02c', alpha=0.9)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('RMSE (log scale)')
ax.set_title('RMSE comparison before/after fixes')
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_dir / 'rmse_before_after_core3.png', dpi=180)
plt.close(fig)

all_n = int(core3_data['events']['all'])
ok_n = int(core3_data['events']['after_filter'])
fail_n = all_n - ok_n

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.pie(
    [ok_n, fail_n],
    labels=[f'stop_id==0 ({ok_n})', f'stop_id>0 ({fail_n})'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['#4daf4a', '#ff7f00'],
)
ax.set_title('Event quality split in source ROOT')
fig.tight_layout()
fig.savefig(out_dir / 'event_quality_split.png', dpi=180)
plt.close(fig)

print('Output dir:', out_dir)
for p in sorted(out_dir.glob('*.png')):
    print('-', p.name)
