# Dental ML Force Prediction

Machine learning models for predicting orthodontic aligner forces across different aligner thicknesses (0.25mm, 0.5mm, 0.75mm, 1.0mm, 1.25mm).

## Background

Orthodontic aligners apply forces and moments to teeth during treatment. This project uses ML to predict those forces at thicknesses where no real experimental data exists, using data from 0.25mm and 0.5mm aligners.

- **Teeth analyzed:** U6, U7
- **Forces & Moments:** Fx, Fy, Fz (N), Tx, Ty, Tz (Nmm)
- **Input data:** `smith_dataset.csv` (DPA-type rows are extracted; see `xgboost/extract_dpa.py`)

## Project Structure

```
dental-ml-force-prediction/
├── correlation/        # Force/moment correlation analysis
├── xgboost/            # XGBoost-based prediction (and linear baseline)
├── gpr/                # Gaussian Process Regression prediction
└── LSTM/               # LSTM cohort prediction with Leave-One-Out CV
```

## Quick Start

```bash
# 0. Install dependencies (Python 3.9 recommended)
pip install -r requirements.txt

# 1. Extract DPA rows from the source dataset
#    (produces dpa_025.csv, dpa_05.csv used by every later script)
cd xgboost && python extract_dpa.py && cd ..

# 2. Correlation matrix between the 6 force/moment components
cd correlation && python correlation.py && cd ..

# 3. XGBoost extrapolation + linear baseline
cd xgboost
python xgboost_force_prediction.py
python linear_extrapolation.py
python xgboost_split_comparison.py

# 4. Generate simulated 0.75mm data (delta-based, w = 0.5 ~ 1.5)
python generate_sim_075.py
python simulation_075.py
python plot_comparison.py
cd ..

# 5. GPR — Step 1: train on real 0.25 + 0.5mm only
cd gpr && python gpr_prediction.py

# 6. GPR — Step 2: add simulated 0.75mm data, predict 1.0mm & 1.25mm
python gpr_with_sim.py

# 7. LSTM — single run (train Cohorts 1-4, predict Cohort 5)
cd LSTM && python lstm_cohort.py

# 8. LSTM — Leave-One-Out CV (5 folds across all cohorts)
python lstm_loo.py
```

> Each script writes its outputs (PNG plots and intermediate CSVs) into the same directory it is run from. Move them into the matching `results/` folder if you want them to render in the per-folder READMEs.

## 1. Correlation Analysis

Pairwise Pearson correlations between all 6 force/moment components on the combined 0.25mm + 0.5mm DPA data.

→ See [`correlation/README.md`](correlation/README.md)

## 2. XGBoost

Trains one XGBoost regressor per force component on 0.25mm + 0.5mm data and uses it to predict 0.75mm and 1.0mm.

**Key finding:** XGBoost (a tree-based model) cannot extrapolate. Predictions at 0.75mm and 1.0mm collapse to identical flat lines — the model just repeats the boundary value for any input outside the training range. This limitation motivated the switch to GPR.

→ See [`xgboost/README.md`](xgboost/README.md)

## 3. GPR — Gaussian Process Regression

GPR with a Matern kernel (ν = 1.5) trained on real (and optionally simulated) data to predict 1.0mm and 1.25mm forces. Implemented with GPyTorch on GPU.

| | Step 1 (real only) | Step 2 (with simulated data) |
|---|---|---|
| **Training data** | Real 0.25 + 0.5mm | Real 0.25 + 0.5mm + sim 0.75mm |
| **Predicts** | 0.75mm, 1.0mm | 1.0mm, 1.25mm |
| **1.0mm vs 1.25mm distinction** | ❌ nearly identical | ✅ different values |
| **Uncertainty representation** | Confidence band (μ ± 2σ) | Multiple prediction lines (w = 0.5 ~ 1.5) |
| **Time trend** | ❌ flat | ❌ still flat |

→ See [`gpr/README.md`](gpr/README.md)

## 4. LSTM — Cohort Prediction (Leave-One-Out CV)

Unlike XGBoost and GPR (which predict across **aligner thicknesses**), the LSTM model predicts across **patient cohorts** — given 4 cohorts' force/moment time-series data, it predicts the 5th unseen cohort.

- **Architecture:** LSTM (128 hidden, 2 layers) → Dense(64) → scalar output
- **Input:** normalized time (0h ~ 336h = 14 days), **Output:** force or moment value
- **Evaluation:** Leave-One-Out Cross-Validation (LOO-CV) — each of the 5 cohorts is left out once
- **Device:** CPU (BioHPC)

| | Single Run | LOO-CV |
|---|---|---|
| **Script** | `lstm_cohort.py` | `lstm_loo.py` |
| **Train** | Cohorts 1–4 | 4 cohorts (rotating) |
| **Predict** | Cohort 5 | Each cohort in turn |
| **Output** | 4 PNGs | 20 PNGs |

**Key finding:** LSTM captures the overall trend of force/moment over time but produces smooth curves that miss individual patient oscillations — expected given the small dataset size (5 cohorts).

→ See [`LSTM/README.md`](LSTM/README.md)

## Environment

- Python 3.9
- PyTorch (CPU for LSTM, GPU for GPR) + GPyTorch
- XGBoost, scikit-learn, pandas, numpy
- matplotlib, seaborn
- GPU: NVIDIA L40S (Libra HPC) for GPR training

See `requirements.txt` for exact package list.

## Limitations & Future Work

- **Time trend stays flat** even in GPR Step 2. The model captures the thickness axis but not the temporal dynamics — adding richer time features or a time-aware kernel is a likely next step.
- **Only 2–3 thickness levels** of real data. The simulated 0.75mm rows are a delta-based interpolation, not true measurements; conclusions at ≥ 1.0mm should be treated as exploratory.
- **U6 and U7 only.** Generalization to other tooth positions is untested.
- **No cross-validation** on the GPR fits — only train/test split is used in XGBoost.
- **LSTM smooth curves** miss individual patient oscillations due to small cohort size (n=5). LOO-CV gives a more robust MAE estimate but the model itself is still limited by data quantity.

## Data

Source rows are filtered from `smith_dataset.csv` (DPA-type only). If you redistribute this repository, please also document the original data source and licensing — the dataset is **not** included in this repo.
