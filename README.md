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

# 7. LSTM — LOO + Partial Observation forecast (dashed-line style)
#    Hides the last k=1,2,3 time points and predicts them; 60 PNGs total
cd LSTM && python lstm_forecast_v2.py
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

## 4. LSTM — LOO + Partial Observation Forecast

Unlike XGBoost and GPR (which predict across **aligner thicknesses**), the LSTM model predicts across **patient cohorts** — given 4 cohorts' force/moment time-series data, it predicts the last k time points of the 5th unseen cohort.

- **Architecture:** LSTM (128 hidden, 2 layers) → Dense(64) → scalar output
- **Input:** normalized time (0h ~ 336h = 14 days), **Output:** force or moment value
- **Evaluation:** Leave-One-Out (LOO) with Partial Observation — each cohort is left out once; the last k time points are additionally hidden
- **Device:** CPU (BioHPC)

| | LOO + Partial Obs |
|---|---|
| **Script** | `lstm_forecast_v2.py` |
| **Train** | 4 cohorts (all pts) + target cohort (first 11-k pts) |
| **Predict** | Last k time points per cohort (k = 1, 2, 3) |
| **Visualization** | Gray solid (full actual) + colored dashed (LSTM prediction) + black triangles (hidden truth) |
| **Output** | 60 PNGs (5 cohorts × 3 k-values × 4 sheets) |

**Partial Observation:** For each target cohort, the last k time points are hidden during training (k=1 hides 14 days; k=2 hides 7 Days + 14 days; k=3 hides 6 Days + 7 Days + 14 days). The LSTM is trained on the remaining data and predicts the hidden points. Visualization shows the full actual time series as a gray solid line, the LSTM prediction as a colored dashed extension from the last observed point, and actual hidden values as black triangles.

**Key finding:** LSTM captures the overall trend of force/moment over time but produces smooth predictions that miss individual patient oscillations — expected given the small dataset size (5 cohorts).

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
