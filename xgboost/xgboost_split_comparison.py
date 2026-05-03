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

# 2. Train/Test Split
X_train_100, y_train_100 = X, y
X_train_80, _, y_train_80, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_70, _, y_train_70, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_50, _, y_train_50, _ = train_test_split(X, y, test_size=0.5, random_state=42)

splits = {
    '100%': (X_train_100, y_train_100),
    '80%' : (X_train_80,  y_train_80),
    '70%' : (X_train_70,  y_train_70),
    '50%' : (X_train_50,  y_train_50),
}

# 3. Train Models
all_models = {}
for label, (X_tr, y_tr) in splits.items():
    print(f"Training {label} — {len(X_tr)} rows")
    models = {}
    for target in targets:
        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr[target])
        models[target] = model
    all_models[label] = models

# 4. Predict at 1.0mm
time_range = np.linspace(df['time_hours'].min(), df['time_hours'].max(), 500)

force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

colors = {'100%': 'black', '80%': 'green', '70%': 'red', '50%': 'purple'}

actual_025 = dpa_025.copy()
actual_025['tooth_num'] = actual_025['tooth'].str.extract(r'(\d+)').astype(int)
actual_05  = dpa_05.copy()
actual_05['tooth_num']  = actual_05['tooth'].str.extract(r'(\d+)').astype(int)

# 5. Plot
for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:

    X_pred = pd.DataFrame({'time_hours': time_range, 'thickness': 1.0, 'tooth_num': tooth_num})

    mean_025 = actual_025[actual_025['tooth_num'] == tooth_num].groupby('time_hours')[targets].mean().reset_index()
    mean_05  = actual_05[actual_05['tooth_num'] == tooth_num].groupby('time_hours')[targets].mean().reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{tooth_label} — Predicted 1.0mm at Different Training Sizes', fontsize=13)

    for ax, target in zip(axes.flatten(), targets):
        ax.plot(mean_025['time_hours'], mean_025[target], color='steelblue',  label='Actual 0.25mm', linewidth=1.5)
        ax.plot(mean_05['time_hours'],  mean_05[target],  color='darkorange', label='Actual 0.5mm',  linewidth=1.5)

        for label, models in all_models.items():
            pred = pd.Series(models[target].predict(X_pred))
            pred = pred.rolling(window=20, center=True, min_periods=1).mean()
            ax.plot(time_range, pred, color=colors[label], label=f'Pred 1.0mm ({label})', linewidth=1.5, linestyle='--')

        ax.set_title(force_labels[target])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(force_labels[target])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'compare_splits_1mm_{tooth_label}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")
