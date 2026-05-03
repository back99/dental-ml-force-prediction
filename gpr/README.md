# GPR — Gaussian Process Regression Force Prediction

Uses Gaussian Process Regression (GPR) to predict orthodontic forces beyond the training data range (extrapolation), as a more principled alternative to tree-based models.

## Why GPR over XGBoost?

| | XGBoost | GPR |
|---|---|---|
| **Extrapolation** | ❌ cannot predict beyond training range | ✅ capable |
| **Uncertainty output** | ❌ | ✅ confidence interval (μ ± 2σ) |
| **Small data** | risk of overfitting | can leverage kernel prior |
| **Speed** | fast | slow, $\mathcal{O}(n^3)$ |

## Model

- **Kernel:** Matern (ν = 1.5)
- **Library:** GPyTorch (GPU-accelerated)
- **Features:** `time_hours`, `thickness`
- **Targets:** `fx`, `fy`, `fz`, `tx`, `ty`, `tz`
- **Training:** 100 Adam iterations, lr = 0.1, marginal log-likelihood loss
- **GPU:** NVIDIA L40S (48GB VRAM) on Libra HPC

## Inputs

- `dpa_025.csv`, `dpa_05.csv` — produced by `../xgboost/extract_dpa.py`
- `sim_075_{U6,U7}_w{w}.csv` — produced by `generate_sim_075.py` (weighted delta simulation)

## Scripts

| File | Description |
|---|---|
| `generate_sim_075.py` | Generates simulated 0.75mm data with weights w = 0.5, 0.8, 1.0, 1.2, 1.5. **Identical to `../xgboost/generate_sim_075.py`** — kept here for convenience; safe to remove and import the other copy. |
| `gpr_prediction.py` | GPR trained on real 0.25 + 0.5mm only → predicts 0.75mm & 1.0mm. Saves `gpr_{tooth}.csv` and `gpr_{tooth}.png`. |
| `gpr_with_sim.py` | GPR trained on real 0.25 + 0.5mm + simulated 0.75mm (one model per weight) → predicts 1.0mm & 1.25mm. Saves `gpr_sim_{tooth}.png` and per-weight CSVs. |

## How to run

```bash
# Prereq: dpa_025.csv and dpa_05.csv must already exist (run ../xgboost/extract_dpa.py first)
cd gpr
python generate_sim_075.py
python gpr_prediction.py     # Step 1
python gpr_with_sim.py       # Step 2
```

## Results

### Step 1 — Real data only (0.25 + 0.5mm → predicts 0.75mm & 1.0mm)

Shaded band = GPR confidence interval (μ ± 2σ), representing **model** uncertainty.

| Tooth U6 | Tooth U7 |
|---|---|
| ![GPR Step 1 — U6](results/gpr_U6.png) | ![GPR Step 1 — U7](results/gpr_U7.png) |

**Finding:** Both predictions are nearly identical flat lines — only 2 thickness points is insufficient for reliable extrapolation.

### Step 2 — With simulated 0.75mm (0.25 + 0.5 + sim 0.75mm → predicts 1.0mm & 1.25mm)

Multiple lines (w = 0.5 ~ 1.5) = one GPR per weight value, representing **data** uncertainty from the simulation.

| Tooth U6 | Tooth U7 |
|---|---|
| ![GPR Step 2 — U6](results/gpr_sim_U6.png) | ![GPR Step 2 — U7](results/gpr_sim_U7.png) |

**Finding:** 1.0mm and 1.25mm are now distinguishable ✅ — but the time trend is still flat ❌ (capturing thickness, not temporal dynamics).
