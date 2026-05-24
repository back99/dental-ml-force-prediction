import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

dpa_025['tooth_num'] = dpa_025['tooth'].str.extract(r'(\d+)').astype(int)
dpa_05['tooth_num']  = dpa_05['tooth'].str.extract(r'(\d+)').astype(int)

targets = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
labels  = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# ─────────────────────────────────────────
# 2. Plot: U6 and U7 x 0.25mm and 0.5mm
#    = 4 subplots in one figure
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Pearson Correlation Matrix — 6 Force Components', fontsize=14)

configs = [
    (dpa_025, 'U6', 6, axes[0][0]),
    (dpa_025, 'U7', 7, axes[0][1]),
    (dpa_05,  'U6', 6, axes[1][0]),
    (dpa_05,  'U7', 7, axes[1][1]),
]

for df, tooth_label, tooth_num, ax in configs:
    subset = df[df['tooth_num'] == tooth_num][targets]
    corr = subset.corr()

    thickness = '0.25mm' if df is dpa_025 else '0.5mm'

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(f'{tooth_label} — {thickness}')

plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_matrix.png")
