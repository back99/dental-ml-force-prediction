# GPR — Gaussian Process Regression Force Prediction

Uses Gaussian Process Regression (GPR) to predict orthodontic forces beyond the training data range (extrapolation).

## Why GPR over XGBoost?
| | XGBoost | GPR |
|---|---------|-----|
| Extrapolation | ❌ Cannot predict beyond training range | ✅ Capable |
| Uncertainty output | ❌ | ✅ Outputs confidence interval (μ ± 2σ) |
| Small data | Risk of overfitting | Can leverage kernel prior knowledge |
| Speed | Fast | Slow O(n³) |

## Model
- **Kernel:** Matern (ν=1.5)
- **Library:** GPyTorch (GPU-accelerated)
- **Features:** `time_hours`, `thickness`
- **Targets:** `fx`, `fy`, `fz`, `tx`, `ty`, `tz`
- **GPU:** NVIDIA L40S (48GB VRAM) on Libra HPC

## Scripts
| File | Description |
|------|-------------|
| `generate_sim_075.py` | Generates simulated 0.75mm data (w = 0.5, 0.8, 1.0, 1.2, 1.5) |
| `gpr_prediction.py` | GPR trained on 0.25 + 0.5mm real data → predicts 0.75mm & 1.0mm |
| `gpr_with_sim.py` | GPR trained on 0.25 + 0.5 + sim 0.75mm → predicts 1.0mm & 1.25mm |

## Experiments

### Step 1 — Real data only (gpr_prediction.py)
- **Training:** Real 0.25mm + Real 0.5mm (full data, no sampling)
- **Predicts:** 0.75mm, 1.0mm
- **Uncertainty:** Shaded band = GPR confidence interval (μ ± 2σ)
- **Result:** Both predictions are nearly identical flat lines — only 2 thickness points is insufficient for extrapolation

### Step 2 — With simulated 0.75mm (gpr_with_sim.py)
- **Training:** Real 0.25mm + Real 0.5mm + Simulated 0.75mm (5 weights)
- **Predicts:** 1.0mm, 1.25mm
- **Uncertainty:** Multiple prediction lines — one per w value, representing data uncertainty from simulation
- **Result:** 1.0mm and 1.25mm are now distinguishable ✅ — but time trend is still flat ❌

## Results
| File | Description |
|------|-------------|
| `results/gpr_U6.png` | GPR prediction (real data only) — Tooth U6 |
| `results/gpr_U7.png` | GPR prediction (real data only) — Tooth U7 |
| `results/gpr_sim_U6.png` | GPR prediction (with sim 0.75mm) — Tooth U6 |
| `results/gpr_sim_U7.png` | GPR prediction (with sim 0.75mm) — Tooth U7 |
