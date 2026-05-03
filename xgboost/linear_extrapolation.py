import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

dpa_025['tooth_num'] = dpa_025['tooth'].str.extract(r'(\d+)').astype(int)
dpa_05['tooth_num']  = dpa_05['tooth'].str.extract(r'(\d+)').astype(int)

targets = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

# 2. Linear Extrapolation per tooth
# formula: f(0.75) = 2 * f(0.5) - f(0.25)
for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:

    mean_025 = dpa_025[dpa_025['tooth_num'] == tooth_num].groupby('time_hours')[targets].mean().reset_index()
    mean_05  = dpa_05[dpa_05['tooth_num'] == tooth_num].groupby('time_hours')[targets].mean().reset_index()

    # Merge on common time points
    merged = pd.merge(mean_025, mean_05, on='time_hours', suffixes=('_025', '_05'))

    # Extrapolate to 0.75mm
    linear_075 = pd.DataFrame({'time_hours': merged['time_hours']})
    for target in targets:
        linear_075[target] = 2 * merged[f'{target}_05'] - merged[f'{target}_025']

    # 3. Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Force Comparison — {tooth_label}  (Actual 0.25 & 0.5mm  vs  Linear Extrapolation 0.75mm)', fontsize=13)

    for ax, target in zip(axes.flatten(), targets):
        ax.plot(mean_025['time_hours'], mean_025[target], color='steelblue',  label='Actual 0.25mm', linewidth=1.5)
        ax.plot(mean_05['time_hours'],  mean_05[target],  color='darkorange', label='Actual 0.5mm',  linewidth=1.5)
        ax.plot(linear_075['time_hours'], linear_075[target], color='green', label='Linear 0.75mm', linewidth=2, linestyle='--')

        ax.set_title(force_labels[target])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(force_labels[target])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'linear_extrapolation_{tooth_label}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")
