"""
LSTM Cohort Prediction
- Train: Cohorts 1-4 (raw data)
- Test:  Cohort 5
- Model: LSTM (time -> force/moment)
- Output: PNG plots + MAE summary

Usage:
    python lstm_cohort.py

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
OUT_DIR  = Path("/home/biohpc/Dental/Cohort/results")
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

        df = df.dropna(how='all')
        df = df.dropna(subset=['time'])
        df = df.reset_index(drop=True)

        print(f"    cohort={cohort} sheet={sheet}: {len(df)} rows loaded")
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
    X_train = X_train[mask]
    y_train = y_train[mask]

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


# ── Analysis for one sheet ────────────────────────────────────────────────────
def run_sheet(sheet: str):
    print(f"\n{'='*55}")
    print(f"  Sheet: {sheet}")
    print(f"{'='*55}")

    train_dfs  = []
    cohort_dfs = {}
    for c in range(1, 5):
        df = load_sheet(c, sheet)
        if not df.empty:
            train_dfs.append(df)
            cohort_dfs[c] = df
    test_df = load_sheet(5, sheet)

    if not train_dfs:
        print(f"  [ERROR] No training data for {sheet}")
        return None

    train_all = pd.concat(train_dfs, ignore_index=True)
    print(f"  Total training rows: {len(train_all)}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{sheet}  |  Train: Cohort 1-4  →  Predict: Cohort 5  (LSTM)",
        fontsize=12, fontweight='bold'
    )
    axes = axes.flatten()

    T_dense = np.linspace(0, MAX_HOURS, 300)
    errors  = {}

    for i, comp in enumerate(COMPONENTS):
        ax    = axes[i]
        color = COMP_COLORS[comp]
        print(f"\n  Training {comp.upper()} ...")

        if comp not in train_all.columns or 'time' not in train_all.columns:
            print(f"    [WARN] Column missing, skipping.")
            continue

        sub  = train_all[['time', comp]].dropna()
        X_tr = sub['time'].values.astype(float) / MAX_HOURS
        y_tr = sub[comp].values.astype(float)

        mu_dense = train_and_predict(X_tr, y_tr, T_dense / MAX_HOURS)

        # Individual cohort lines (gray)
        for c, cdf in cohort_dfs.items():
            if 'time' not in cdf.columns or comp not in cdf.columns:
                continue
            cmean = (cdf.groupby('time')[comp].mean()
                       .reset_index()
                       .sort_values('time'))
            ax.plot(cmean['time'], cmean[comp],
                    color='#bbbbbb', linewidth=1.0, alpha=0.7, zorder=1)

        ax.plot(T_dense, mu_dense, color=color,
                linewidth=2.0, zorder=3, label='LSTM pred')

        mae = float('nan')
        if not test_df.empty and 'time' in test_df.columns and comp in test_df.columns:
            tmean = (test_df.groupby('time')[comp].mean()
                           .reset_index()
                           .sort_values('time'))
            X_te = tmean['time'].values.astype(float)
            y_te = tmean[comp].values.astype(float)

            ax.plot(X_te, y_te, 'k^--', markersize=7,
                    linewidth=1.5, zorder=5, label='Cohort 5 actual')
            ax.scatter(X_te, y_te, color='black', s=50, zorder=6)

            mu_te = train_and_predict(X_tr, y_tr, X_te / MAX_HOURS)
            mae   = float(np.nanmean(np.abs(mu_te - y_te)))

        errors[comp] = mae
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
    out_path = OUT_DIR / f"lstm_{sheet}.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    print(f"  MAE Summary:")
    for comp, e in errors.items():
        unit = "N" if comp.startswith('f') else "N·mm"
        print(f"    {comp.upper()}: {e:.4f} {unit}")

    return errors


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LSTM Cohort Prediction — Train Cohorts 1-4, Predict Cohort 5")

    all_errors = {}
    for sheet in SHEETS:
        result = run_sheet(sheet)
        if result:
            all_errors[sheet] = result

    print("\n" + "="*55)
    print("=== FINAL MAE SUMMARY ===")
    print("="*55)
    for sheet, errs in all_errors.items():
        print(f"\n{sheet}:")
        for comp, e in errs.items():
            unit = "N" if comp.startswith('f') else "N·mm"
            print(f"  {comp.upper()}: {e:.4f} {unit}")
