import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

df = pd.concat([dpa_025, dpa_05], ignore_index=True)
df['tooth_num'] = df['tooth'].str.extract(r'(\d+)').astype(int)

dpa_025['tooth_num'] = dpa_025['tooth'].str.extract(r'(\d+)').astype(int)
dpa_05['tooth_num']  = dpa_05['tooth'].str.extract(r'(\d+)').astype(int)

features = ['time_hours', 'thickness', 'tooth_num']
targets  = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

X = df[features]
y = df[targets]

# ─────────────────────────────────────────
# 2. Train XGBoost Models
# ─────────────────────────────────────────
print("Training XGBoost models...")
models = {}
for target in targets:
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         random_state=42, verbosity=0)
    model.fit(X, y[target])
    models[target] = model
print("Training complete.\n")

weights = [0.5, 0.8, 1.0, 1.2, 1.5]

force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:
    print(f"Processing {tooth_label}...")

    mean_025 = (dpa_025[dpa_025['tooth_num'] == tooth_num]
                .groupby('time_hours')[targets].mean().reset_index())
    mean_05  = (dpa_05[dpa_05['tooth_num'] == tooth_num]
                .groupby('time_hours')[targets].mean().reset_index())

    merged = pd.merge(mean_025, mean_05, on='time_hours', suffixes=('_025', '_05'))
    time_common = merged['time_hours'].values

    # XGBoost prediction at 0.75mm (smoothed)
    X_pred = pd.DataFrame({
        'time_hours': time_common,
        'thickness' : 0.75,
        'tooth_num' : tooth_num
    })
    pred_075 = {}
    for target in targets:
        raw = models[target].predict(X_pred)
        s = pd.Series(raw).rolling(window=20, center=True, min_periods=1).mean()
        pred_075[target] = s.values

    # delta 계산
    delta = {}
    for target in targets:
        delta[target] = merged[f'{target}_05'].values - merged[f'{target}_025'].values

    # ─────────────────────────────────────────
    # 3. w별로 CSV 따로 저장
    # ─────────────────────────────────────────
    sim_results = {}
    for w in weights:
        w_str = str(w).replace('.', '')   # 0.5 → 05
        csv_data = {'time_hours': time_common}

        for target in targets:
            f_05 = merged[f'{target}_05'].values
            sim_val = f_05 + w * delta[target]
            sim_results.setdefault(target, {})[w] = sim_val

            csv_data[f'{target}_actual_025'] = merged[f'{target}_025'].values
            csv_data[f'{target}_actual_05']  = f_05
            csv_data[f'{target}_sim_075']    = sim_val
            csv_data[f'{target}_xgb_075']    = pred_075[target]

        csv_df = pd.DataFrame(csv_data)
        csv_filename = f'sim_075_{tooth_label}_w{w_str}.csv'
        csv_df.to_csv(csv_filename, index=False)
        print(f"  Saved: {csv_filename}  ({len(csv_df)} rows)")

    # ─────────────────────────────────────────
    # 4. Plot
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Simulation Validation — {tooth_label}\n'
        f'Simulated 0.75mm Range (w=0.5~1.5)  vs  XGBoost Predicted 0.75mm',
        fontsize=13
    )

    for ax, target in zip(axes.flatten(), targets):
        sim_stack = np.array([sim_results[target][w] for w in weights])
        sim_min = sim_stack.min(axis=0)
        sim_max = sim_stack.max(axis=0)

        ax.fill_between(time_common, sim_min, sim_max,
                        color='lightgray', alpha=0.7, label='Sim range (w=0.5~1.5)')
        for w in weights:
            ax.plot(time_common, sim_results[target][w],
                    color='gray', linewidth=0.8, alpha=0.5)

        ax.plot(merged['time_hours'], merged[f'{target}_025'],
                color='steelblue', linewidth=1.5, label='Actual 0.25mm')
        ax.plot(merged['time_hours'], merged[f'{target}_05'],
                color='darkorange', linewidth=1.5, label='Actual 0.5mm')
        ax.plot(time_common, pred_075[target],
                color='green', linewidth=2, linestyle='--', label='XGBoost 0.75mm')

        ax.set_title(force_labels[target])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(force_labels[target])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_filename = f'simulation_{tooth_label}.png'
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_filename}\n")

print("Done!")
print("Generated files:")
print("  CSV : sim_075_U6_w05.csv ~ sim_075_U6_w15.csv  (5개)")
print("  CSV : sim_075_U7_w05.csv ~ sim_075_U7_w15.csv  (5개)")
print("  PNG : simulation_U6.png, simulation_U7.png      (2개)")
