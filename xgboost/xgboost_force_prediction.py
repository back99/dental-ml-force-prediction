import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
# Load DPA 0.25mm and 0.5mm activation datasets
dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

# Combine both activation levels — model needs to learn the relationship between thickness and force
df = pd.concat([dpa_025, dpa_05], ignore_index=True)
print(f"Total rows: {len(df)}")

# ─────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────
# Encode tooth as numeric (U6 → 6, U7 → 7) because XGBoost needs numbers
df['tooth_num'] = df['tooth'].str.extract(r'(\d+)').astype(int)

# Input features: time, activation level, tooth
features = ['time_hours', 'thickness', 'tooth_num']

# Output targets: all 6 force components
targets = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

X = df[features]
y = df[targets]

# ─────────────────────────────────────────
# 3. Train / Test Split (80% train, 20% test)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ─────────────────────────────────────────
# 4. Train One XGBoost Model Per Force Component
# ─────────────────────────────────────────
models = {}

print("\n--- Model Performance on Test Set ---")
for target in targets:
    model = XGBRegressor(
        n_estimators=300,    # number of trees
        max_depth=6,         # maximum depth of each tree
        learning_rate=0.05,  # step size per iteration
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train[target])
    models[target] = model

    # Evaluate on test set
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
    print(f"  {target.upper():>4s}  →  R²: {r2:.4f}  |  RMSE: {rmse:.4f}")

# ─────────────────────────────────────────
# 5. Predict Forces at 0.75mm Activation
# ─────────────────────────────────────────
# Generate time points across the full observed range
time_range = np.linspace(df['time_hours'].min(), df['time_hours'].max(), 500)

predictions = {}

for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:
    # Build prediction input with thickness = 0.75 (extrapolation target)
    X_pred = pd.DataFrame({
        'time_hours': time_range,
        'thickness' : 0.75,
        'tooth_num' : tooth_num
    })

    # Predict all 6 force components
    pred_df = pd.DataFrame({'time_hours': time_range})
    for target in targets:
        pred_df[target] = models[target].predict(X_pred)

    predictions[tooth_label] = pred_df

# ─────────────────────────────────────────
# 6. Plot Results
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Predicted Forces at 0.75mm Activation (DPA) — XGBoost Extrapolation', fontsize=13)

force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

for ax, target in zip(axes.flatten(), targets):
    for tooth_label in ['U6', 'U7']:
        pred = predictions[tooth_label]
        ax.plot(pred['time_hours'], pred[target], label=f'{tooth_label} @ 0.75mm')

    ax.set_title(force_labels[target])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel(force_labels[target])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predicted_075mm_forces.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: predicted_075mm_forces.png")
