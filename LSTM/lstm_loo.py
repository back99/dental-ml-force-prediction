"""
LSTM Leave-One-Out Cross-Validation (LOO-CV)
- For each cohort i (1~5): train on remaining 4, predict cohort i
- Repeated 5 times → more reliable evaluation than single train/test split
- Output: PNG per (sheet × left-out cohort) = 20 plots + overall MAE summary

Usage:
    python lstm_loo.py

Data layout:
    data/cohort_1.xlsx ~ cohort_5.xlsx
    Each file has 4 sheets: DPA025_U6, DPA025_U7, DPA05_U6, DPA05_U7
    Each sheet: time (string labels), fx, fy, fz, tx, ty, tz
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
OUT_DIR  = Path("/home/biohpc/Dental/Cohort/results/loo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIME_HOURS = {
    "0h": 0, "8h": 8, "16h": 16, "24h": 24, "48h": 48,
    "3 Days": 72, "4 Days": 96, "5 Days": 120,
    "6 Days": 144, "7 Days": 168, "14 days": 336,
}
TIME_ORDER = list(TIME_HOURS.keys())
T_HOURS    = np.array([TIME_HOURS[t] for t in TIME_ORDER], dtype=float)
MAX_HOURS  = 336.0

SHEETS     = ['DPA025_U6', 'DPA025_U7', 'DPA05_U6', 'DPA05_U7']
COHORTS    = [1, 2, 3, 4, 5]
COMPONENTS = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
COMP_COLORS = {
    'fx': '#e74c3c', 'fy': '#2ecc71', 'fz': '#3498db',
    'tx': '#e67e22', 'ty': '#9b59b6', 'tz': '#1abc9c',
}

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

        # Convert time column (string labels e.g. "0h", "8h") to hours
        # IMPORTANT: must be done BEFORE pd.to_numeric to avoid NaN
        if 'time' in df.columns:
            df['time'] = df['time'].apply(
                lambda x: TIME_HOURS.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan
            )

        # Convert all other columns to numeric
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
    """
    Input:  (batch, seq_len=1, 1) — normalized time value
    Output: (batch,)              — predicted force or moment
    """
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
def train_and_predict(X_train: np.ndarray, y_train: np.ndarray,
                      X_pred: np.ndarray) -> np.ndarray:
    # Remove NaN samples
    mask = ~(np.isnan(X_train) | np.isnan(y_train))
    X_train, y_train = X_train[mask], y_train[mask]

    if len(X_train) == 0:
        print("    [WARN] No valid training data, returning zeros")
        return np.zeros(len(X_pred))

    # Normalize targets
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
            avg = epoch_loss / len(loader)
            print(f"      epoch {epoch+1:3d}/{N_EPOCHS}  loss={avg:.6f}")

    model.eval()
    with torch.no_grad():
        X_q = torch.FloatTensor(X_pred).view(-1, 1, 1).to(device)
        preds = model(X_q).cpu().numpy()

    return preds * y_std + y_mean


# ── LOO for one sheet ─────────────────────────────────────────────────────────
def run_loo_sheet(sheet: str):
    print(f"\n{'='*55}")
    print(f"  Sheet: {sheet}  (Leave-One-Out)")
    print(f"{'='*55}")

    # Load all cohort data
    all_data = {}
    for c in COHORTS:
        df = load_sheet(c, sheet)
        if not df.empty:
            all_data[c] = df
        else:
            print(f"  [WARN] Cohort {c} empty, skipping")

    if len(all_data) < 2:
        print(f"  [ERROR] Not enough data for {sheet}")
        return None

    sheet_errors = {}  # {left_out_cohort: {comp: mae}}
    T_dense = np.linspace(0, MAX_HOURS, 300)

    for test_cohort in COHORTS:
        if test_cohort not in all_data:
            print(f"  [SKIP] Cohort {test_cohort} not available")
            continue

        train_cohorts = [c for c in COHORTS if c != test_cohort and c in all_data]
        print(f"\n  LOO: Train={train_cohorts} → Predict Cohort {test_cohort}")

        train_df = pd.concat([all_data[c] for c in train_cohorts], ignore_index=True)
        test_df  = all_data[test_cohort]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            f"{sheet}  |  Train: Cohort {train_cohorts}  →  Predict: Cohort {test_cohort}  (LSTM LOO)",
            fontsize=11, fontweight='bold'
        )
        axes = axes.flatten()

        comp_errors = {}

        for i, comp in enumerate(COMPONENTS):
            ax    = axes[i]
            color = COMP_COLORS[comp]
            print(f"    Training {comp.upper()} ...")

            if comp not in train_df.columns or 'time' not in train_df.columns:
                continue

            # Training data
            sub  = train_df[['time', comp]].dropna()
            X_tr = sub['time'].values.astype(float) / MAX_HOURS
            y_tr = sub[comp].values.astype(float)

            # Predict dense curve for smooth visualization
            mu_dense = train_and_predict(X_tr, y_tr, T_dense / MAX_HOURS)

            # Training cohort lines (gray background)
            for c in train_cohorts:
                cdf = all_data[c]
                if 'time' not in cdf.columns or comp not in cdf.columns:
                    continue
                cmean = (cdf.groupby('time')[comp].mean()
                           .reset_index().sort_values('time'))
                ax.plot(cmean['time'], cmean[comp],
                        color='#bbbbbb', linewidth=1.0, alpha=0.7, zorder=1)

            # LSTM prediction curve
            ax.plot(T_dense, mu_dense, color=color,
                    linewidth=2.0, zorder=3, label='LSTM pred')

            # Test cohort actual values
            mae = float('nan')
            if 'time' in test_df.columns and comp in test_df.columns:
                tmean = (test_df.groupby('time')[comp].mean()
                               .reset_index().sort_values('time'))
                X_te = tmean['time'].values.astype(float)
                y_te = tmean[comp].values.astype(float)

                ax.plot(X_te, y_te, 'k^--', markersize=7,
                        linewidth=1.5, zorder=5,
                        label=f'Cohort {test_cohort} actual')
                ax.scatter(X_te, y_te, color='black', s=50, zorder=6)

                mu_te = train_and_predict(X_tr, y_tr, X_te / MAX_HOURS)
                mae   = float(np.nanmean(np.abs(mu_te - y_te)))

            comp_errors[comp] = mae
            unit = "N" if comp.startswith('f') else "N·mm"
            ax.set_title(f"{comp.upper()}  MAE={mae:.4f} {unit}", fontsize=10)
            ax.set_xlabel("Hours", fontsize=8)
            ax.set_ylabel(unit, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='best')
            ax.axhline(0, color='#dddddd', linewidth=0.5, linestyle='--')
            ax.set_xticks(T_HOURS)
            ax.set_xticklabels(TIME_ORDER, rotation=45, ha='right', fontsize=6)

        plt.tight_layout()
        out_path = OUT_DIR / f"lstm_loo_{sheet}_predict_cohort{test_cohort}.png"
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {out_path}")

        sheet_errors[test_cohort] = comp_errors

    return sheet_errors


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LSTM Leave-One-Out Cross-Validation")
    print(f"Output: {OUT_DIR}")

    all_results = {}
    for sheet in SHEETS:
        result = run_loo_sheet(sheet)
        if result:
            all_results[sheet] = result

    # Print full MAE summary table
    print("\n" + "="*60)
    print("=== LOO MAE SUMMARY ===")
    print("="*60)
    for sheet, cohort_errs in all_results.items():
        print(f"\n[{sheet}]")
        print(f"  {'Cohort':<10}", end="")
        for comp in COMPONENTS:
            unit = "N" if comp.startswith('f') else "N·mm"
            print(f"  {comp.upper()}({unit})", end="")
        print()
        for test_c, comp_errs in sorted(cohort_errs.items()):
            print(f"  Cohort {test_c:<4}", end="")
            for comp in COMPONENTS:
                e = comp_errs.get(comp, float('nan'))
                print(f"  {e:>10.4f}", end="")
            print()

        # Average MAE across all left-out cohorts
        print(f"  {'Average':<10}", end="")
        for comp in COMPONENTS:
            vals = [cohort_errs[c].get(comp, np.nan)
                    for c in cohort_errs if comp in cohort_errs[c]]
            print(f"  {np.nanmean(vals):>10.4f}", end="")
        print()
