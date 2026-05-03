import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load Data
dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

df = pd.concat([dpa_025, dpa_05], ignore_index=True)
df['tooth_num'] = df['tooth'].str.extract(r'(\d+)').astype(int)

features = ['time_hours', 'thickness', 'tooth_num']
targets  = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

X = df[features]
y = df[targets]

# 2. Train Models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {}
for target in targets:
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X_train, y_train[target])
    models[target] = model

# 3. Actual mean values
actual_025 = dpa_025.copy()
actual_025['tooth_num'] = actual_025['tooth'].str.extract(r'(\d+)').astype(int)

actual_05 = dpa_05.copy()
actual_05['tooth_num'] = actual_05['tooth'].str.extract(r'(\d+)').astype(int)

time_range = np.linspace(df['time_hours'].min(), df['time_hours'].max(), 500)

force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:

    mean_025 = actual_025[actual_025['tooth_num'] == tooth_num].groupby('time_hours')[targets].mean().reset_index()
    mean_05  = actual_05[actual_05['tooth_num'] == tooth_num].groupby('time_hours')[targets].mean().reset_index()

    # Predict at 0.75mm
    X_pred = pd.DataFrame({'time_hours': time_range, 'thickness': 0.75, 'tooth_num': tooth_num})
    pred_075 = pd.DataFrame({'time_hours': time_range})
    for target in targets:
        pred_075[target] = models[target].predict(X_pred)
        # Smoothing
        pred_075[target] = pred_075[target].rolling(window=20, center=True, min_periods=1).mean()

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Force Comparison — {tooth_label}  (Actual 0.25 & 0.5mm  vs  Predicted 0.75mm)', fontsize=13)

    for ax, target in zip(axes.flatten(), targets):
        ax.plot(mean_025['time_hours'], mean_025[target], color='steelblue',  label='Actual 0.25mm', linewidth=1.5)
        ax.plot(mean_05['time_hours'],  mean_05[target],  color='darkorange', label='Actual 0.5mm',  linewidth=1.5)
        ax.plot(pred_075['time_hours'], pred_075[target], color='green',      label='Predicted 0.75mm', linewidth=2, linestyle='--')

        ax.set_title(force_labels[target])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(force_labels[target])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'comparison_{tooth_label}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")
