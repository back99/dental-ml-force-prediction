"""
LSTM Time-Series Forecasting — LOO + Partial Observation  (v2: dashed-line style)
==================================================================================
For each target cohort N (1~5):
  - Train: remaining 4 cohorts (all time points) + cohort N (first 11-k time points)
  - Predict: cohort N's last k time points  (k = 1, 2, 3)

Visualization style (v2):
  - Gray solid line : target cohort observed portion
  - Colored dashed  : dashed extension from last observed point → LSTM predicted test points
  - Colored circles : LSTM predicted values at test time points
  - Black triangles : actual (hidden) ground-truth test values

Output: 5 cohorts × 3 k-values × 4 sheets = 60 PNG plots → forecast_v2/

Usage:
    python lstm_forecast_v2.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/biohpc/Dental/Cohort/data")
OUT_DIR  = Path("/home/biohpc/Dental/Cohort/results/forecast_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIME_HOURS = {
    "0h": 0, "8h": 8, "16h": 16, "24h": 24, "48h": 48,
    "3 Days": 72, "4 Days": 96, "5 Days": 120,
    "6 Days": 144, "7 Days": 168, "14 days": 336,
}
TIME_ORDER = list(TIME_HOURS.keys())   # 11 time points in order
T_HOURS    = np.array([TIME_HOURS[t] for t in TIME_ORDER], dtype=float)
MAX_HOURS  = 336.0

SHEETS     = ['DPA025_U6', 'DPA025_U7', 'DPA05_U6', 'DPA05_U7']
COHORTS    = [1, 2, 3, 4, 5]
COMPONENTS = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
COMP_COLORS = {
    'fx': '#e74c3c', 'fy': '#2ecc71', 'fz': '#3498db',
    'tx': '#e67e22', 'ty': '#9b59b6', 'tz': '#1abc9c',
}

K_VALUES = [1, 2, 3]   # number of final time points to predict

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
N_EPOCHS    = 300
LR          = 1e-3
BATCH_SIZE  = 512

device = torch.device('cpu')
print(f"Using device: {device}")

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_sheet(cohort: int, sheet: str) -> pd.DataFrame:
    fp = DATA_DIR / f"cohort_{cohort}.xlsx"
    try:
        df = pd.read_excel(fp, sheet_name=sheet, engine='openpyxl')
        if 'time' in df.columns:
            df['time'] = df['time'].apply(
                lambda x: TIME_HOURS.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan
            )
        for col in df.columns:
            if col != 'time':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(how='all').dropna(subset=['time']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [WARN] Load failed cohort={cohort} sheet={sheet}: {e}")
        return pd.DataFrame()


# ── LSTM Model ────────────────────────────────────────────────────────────────
class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Train & Predict ───────────────────────────────────────────────────────────
def train_and_predict(X_train, y_train, X_pred):
    mask = ~(np.isnan(X_train) | np.isnan(y_train))
    X_train, y_train = X_train[mask], y_train[mask]

    if len(X_train) == 0:
        return np.zeros(len(X_pred))

    y_mean = float(y_train.mean())
    y_std  = float(y_train.std())
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    X_t = torch.FloatTensor(X_train).view(-1, 1, 1)
    y_t = torch.FloatTensor(y_norm)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=min(BATCH_SIZE, len(X_train)), shuffle=True)

    model     = LSTMRegressor(HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"      epoch {epoch+1:3d}/{N_EPOCHS}  loss={epoch_loss/len(loader):.6f}")

    model.eval()
    with torch.no_grad():
        X_q = torch.FloatTensor(X_pred).view(-1, 1, 1).to(device)
        preds = model(X_q).cpu().numpy()

    return preds * y_std + y_mean


# ── Run one cohort × one k ────────────────────────────────────────────────────
def run_forecast(sheet, target_cohort, k, all_data):
    """
    target_cohort : the cohort being predicted (1~5)
    k             : number of final time points to hide and predict (1, 2, or 3)

    Train = other 4 cohorts (all 11 pts) + target cohort (first 11-k pts)
    Test  = target cohort (last k pts)

    Visualization (v2 dashed style):
      Gray solid  : observed time series of target cohort
      Colored dash: dashed extension from last observed → LSTM predicted test points
      Colored dot : LSTM predicted values
      Black ^     : actual (hidden) ground-truth
    """
    train_times   = T_HOURS[:11 - k]
    test_times    = T_HOURS[11 - k:]
    other_cohorts = [c for c in COHORTS if c != target_cohort and c in all_data]

    # ── Skip if output already exists ────────────────────────────────────────
    out_path = OUT_DIR / f"lstm_forecast_v2_{sheet}_cohort{target_cohort}_k{k}.png"
    if out_path.exists():
        print(f"\n  [Cohort {target_cohort}, k={k}]  SKIP (already exists: {out_path.name})")
        return None

    print(f"\n  [Cohort {target_cohort}, k={k}]  train on {other_cohorts} + Cohort {target_cohort} first {11-k} pts")
    print(f"    Hidden: {[TIME_ORDER[i] for i in range(11-k, 11)]}")

    # ── Build training set ────────────────────────────────────────────────────
    train_parts = [all_data[c] for c in other_cohorts]
    if target_cohort in all_data:
        c_train = all_data[target_cohort][all_data[target_cohort]['time'].isin(train_times)]
        train_parts.append(c_train)

    if not train_parts:
        return None

    train_df = pd.concat(train_parts, ignore_index=True)

    # ── Test set ──────────────────────────────────────────────────────────────
    test_df = pd.DataFrame()
    if target_cohort in all_data:
        test_df = all_data[target_cohort][all_data[target_cohort]['time'].isin(test_times)].copy()

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{sheet}  |  Target: Cohort {target_cohort}, k={k} (predict last {k} time point(s))\n"
        f"Train: Cohorts {other_cohorts} (all) + Cohort {target_cohort} (first {11-k}/11 pts)",
        fontsize=10, fontweight='bold'
    )
    axes = axes.flatten()

    comp_errors = {}

    for i, comp in enumerate(COMPONENTS):
        ax    = axes[i]
        color = COMP_COLORS[comp]
        print(f"    {comp.upper()} ...", end=' ', flush=True)

        if comp not in train_df.columns:
            print("skip")
            continue

        sub  = train_df[['time', comp]].dropna()
        X_tr = sub['time'].values.astype(float) / MAX_HOURS
        y_tr = sub[comp].values.astype(float)

        # ── Gray solid: ALL actual data of target cohort (train + test) ──────
        obs_mean = None
        if target_cohort in all_data:
            c_all = all_data[target_cohort]
            if not c_all.empty and comp in c_all.columns:
                all_mean = (c_all.groupby('time')[comp]
                               .mean().reset_index().sort_values('time'))
                ax.plot(all_mean['time'], all_mean[comp],
                        'o-', color='#888888', linewidth=1.5, markersize=6,
                        zorder=2, label=f'Cohort {target_cohort} actual')
            # Also keep observed-only mean for dashed line anchor
            c_obs = c_all[c_all['time'].isin(train_times)]
            if not c_obs.empty and comp in c_obs.columns:
                obs_mean = (c_obs.groupby('time')[comp]
                               .mean().reset_index().sort_values('time'))

        # ── Predict test points & draw dashed extension ───────────────────
        mae = float('nan')
        if not test_df.empty and comp in test_df.columns:
            test_mean = (test_df.groupby('time')[comp]
                                .mean().reset_index().sort_values('time'))
            X_te = test_mean['time'].values.astype(float)
            y_te = test_mean[comp].values.astype(float)

            mu_te = train_and_predict(X_tr, y_tr, X_te / MAX_HOURS)
            mae   = float(np.nanmean(np.abs(mu_te - y_te)))

            # Dashed extension: last training point → LSTM predicted test points
            if obs_mean is not None and not obs_mean.empty:
                last_t = float(obs_mean['time'].iloc[-1])
                last_v = float(obs_mean[comp].iloc[-1])
                dash_t = np.concatenate([[last_t], X_te])
                dash_v = np.concatenate([[last_v], mu_te])
                ax.plot(dash_t, dash_v, '--', color=color, linewidth=2.5,
                        zorder=4, label='LSTM prediction')
            else:
                ax.plot(X_te, mu_te, '--', color=color, linewidth=2.5,
                        zorder=4, label='LSTM prediction')

            # Colored circles at LSTM predicted test points
            ax.scatter(X_te, mu_te, color=color, s=120,
                       edgecolors='white', linewidths=1.5, zorder=5)

            # Black triangles: actual hidden values (overlaid on solid line)
            ax.plot(X_te, y_te, 'k^', markersize=10, zorder=6,
                    label='Actual (hidden)')

        comp_errors[comp] = mae
        unit = "N" if comp.startswith('f') else "N·mm"
        ax.set_title(f"{comp.upper()}  MAE={mae:.4f} {unit}", fontsize=10)
        ax.set_xlabel("Hours", fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='best')
        ax.axhline(0, color='#dddddd', linewidth=0.5, linestyle='--')
        if len(train_times) > 0:
            ax.axvline(x=train_times[-1], color='#aaaaaa',
                       linewidth=1.0, linestyle=':', alpha=0.8)
        ax.set_xticks(T_HOURS)
        ax.set_xticklabels(TIME_ORDER, rotation=45, ha='right', fontsize=6)
        print(f"MAE={mae:.4f}")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")

    return comp_errors


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LSTM Forecast v2 — LOO + Partial Observation (dashed-line style)")
    print(f"Cohorts: {COHORTS}  |  k={K_VALUES}  |  Sheets: {SHEETS}")
    print(f"Total plots: {len(COHORTS)} × {len(K_VALUES)} × {len(SHEETS)} = {len(COHORTS)*len(K_VALUES)*len(SHEETS)}")
    print(f"Output: {OUT_DIR}\n")

    # all_results[sheet][target_cohort][k] = {comp: mae}
    all_results = {}

    for sheet in SHEETS:
        print(f"\n{'='*60}")
        print(f"  Sheet: {sheet}")
        print(f"{'='*60}")

        # Load all cohorts once per sheet
        all_data = {}
        for c in COHORTS:
            df = load_sheet(c, sheet)
            if not df.empty:
                all_data[c] = df

        sheet_results = {}
        for target_cohort in COHORTS:
            cohort_results = {}
            for k in K_VALUES:
                result = run_forecast(sheet, target_cohort, k, all_data)
                if result:
                    cohort_results[k] = result
            sheet_results[target_cohort] = cohort_results

        all_results[sheet] = sheet_results

    # ── MAE Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("=== FORECAST v2 MAE SUMMARY ===")
    print("="*60)
    for sheet, cohort_results in all_results.items():
        print(f"\n[{sheet}]")
        for k in K_VALUES:
            print(f"\n  k={k} (predict last {k} time point(s)):")
            print(f"  {'Cohort':<10}", end="")
            for comp in COMPONENTS:
                unit = "N" if comp.startswith('f') else "N·mm"
                print(f"  {comp.upper()}({unit})", end="")
            print()
            maes = {comp: [] for comp in COMPONENTS}
            for target_cohort in COHORTS:
                errs = cohort_results.get(target_cohort, {}).get(k, {})
                print(f"  Cohort {target_cohort:<4}", end="")
                for comp in COMPONENTS:
                    e = errs.get(comp, float('nan'))
                    maes[comp].append(e)
                    print(f"  {e:>10.4f}", end="")
                print()
            print(f"  {'Average':<10}", end="")
            for comp in COMPONENTS:
                print(f"  {np.nanmean(maes[comp]):>10.4f}", end="")
            print()
