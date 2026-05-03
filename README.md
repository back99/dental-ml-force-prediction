# Dental ML Force Prediction

Machine learning models for predicting orthodontic aligner forces across different aligner thicknesses (0.25mm, 0.5mm, 0.75mm, 1.0mm, 1.25mm).

## Background

Orthodontic aligners apply forces and moments to teeth during treatment. This project uses ML to predict those forces at thicknesses where no real experimental data exists, using data from 0.25mm and 0.5mm aligners.

**Teeth analyzed:** U6, U7
**Forces & Moments:** Fx, Fy, Fz (N), Tx, Ty, Tz (Nmm)

---

## Project Structure

```
dental-ml-force-prediction/
├── correlation/        # Force correlation analysis
├── xgboost/           # XGBoost-based prediction
└── gpr/               # Gaussian Process Regression prediction
```

---

## Modules

### 1. Correlation Analysis (`correlation/`)
Analyzes pairwise correlations between all 6 force/moment components across aligner thicknesses.

→ See [`correlation/README.md`](correlation/README.md)

---

### 2. XGBoost (`xgboost/`)
Trains XGBoost on 0.25mm + 0.5mm data to predict forces at 0.75mm.

Also generates **simulated 0.75mm data** using a weighted delta method:

$$f_{sim}(0.75mm) = f_{individual}(0.5mm) + w \times \delta(t)$$

**Key finding:** XGBoost cannot extrapolate beyond its training range — predictions at 0.75mm and 1.0mm are identical flat lines. This motivated the switch to GPR.

→ See [`xgboost/README.md`](xgboost/README.md)

---

### 3. GPR — Gaussian Process Regression (`gpr/`)
Uses GPR with Matern kernel (ν=1.5) to extrapolate forces beyond the training data range.

Two experiments:
- **Step 1:** Trained on real 0.25 + 0.5mm → predicts 0.75mm & 1.0mm
- **Step 2:** Trained on real 0.25 + 0.5mm + simulated 0.75mm → predicts 1.0mm & 1.25mm

| | Step 1 | Step 2 |
|---|--------|--------|
| 1.0mm vs 1.25mm distinction | ❌ Nearly identical | ✅ Different values |
| Uncertainty representation | Confidence band (μ ± 2σ) | Multiple prediction lines (w = 0.5~1.5) |
| Time trend | ❌ Flat | ❌ Still flat |

→ See [`gpr/README.md`](gpr/README.md)

---

## Data
Real experimental data (CSV) is not included in this repository.
- `dpa_025.csv` — 0.25mm aligner measurements
- `dpa_05.csv` — 0.5mm aligner measurements

## Environment
- Python 3.9
- PyTorch + GPyTorch (GPU)
- XGBoost, scikit-learn, pandas, matplotlib
- GPU: NVIDIA L40S (Libra HPC) for GPR training
