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

## Results

### Step 1 — Real data only (0.25 + 0.5mm → predicts 0.75mm & 1.0mm)

> Shaded band = GPR confidence interval (μ ± 2σ) — represents **model uncertainty**

#### Tooth U6
![GPR U6](results/gpr_U6.png)

#### Tooth U7
![GPR U7](results/gpr_U7.png)

**Finding:** Both predictions are nearly identical flat lines — only 2 thickness points is insufficient for extrapolation.

---

### Step 2 — With simulated 0.75mm (0.25 + 0.5 + sim 0.75mm → predicts 1.0mm & 1.25mm)

> Multiple lines (w = 0.5~1.5) = one GPR per weight value — represents **data uncertainty** from simulation

#### Tooth U6
![GPR Sim U6](results/gpr_sim_U6.png)

#### Tooth U7
![GPR Sim U7](results/gpr_sim_U7.png)

**Finding:** 1.0mm and 1.25mm are now distinguishable ✅ — but time trend is still flat ❌
