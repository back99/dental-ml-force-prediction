# LSTM LOO + Partial Observation Forecast

LSTM (Long Short-Term Memory) model for forecasting dental aligner force/moment time series using Leave-One-Out cross-validation with partial observation.

## Files

| File | Description |
|------|-------------|
| `lstm_forecast_v2.py` | LOO + Partial Observation forecast with dashed-line visualization |

## Data

- `data/cohort_1.xlsx` ~ `cohort_5.xlsx`
- 4 sheets per file: `DPA025_U6`, `DPA025_U7`, `DPA05_U6`, `DPA05_U7`
- Columns: `time` (string labels: "0h", "8h", ..., "14 days"), `fx`, `fy`, `fz`, `tx`, `ty`, `tz`
- 11 time points: 0h, 8h, 16h, 24h, 48h, 3 Days, 4 Days, 5 Days, 6 Days, 7 Days, 14 days

## Model

```
Input: normalized time (scalar) → LSTM(128 hidden, 2 layers) → Dense(64) → output (scalar)
```

- Hidden size: 128, Layers: 2, Dropout: 0.2
- Epochs: 300, LR: 1e-3 (StepLR ×0.5 every 100 epochs), Batch: 512
- Device: CPU

## Experiment Design

For each target cohort N (1~5) and each k ∈ {1, 2, 3}:

- **Train:** remaining 4 cohorts (all 11 time points) + cohort N (first 11-k time points)
- **Predict:** cohort N's last k time points

| k | Hidden time points |
|---|---|
| 1 | 14 days |
| 2 | 7 Days, 14 days |
| 3 | 6 Days, 7 Days, 14 days |

## Visualization Style

Each plot shows 6 subplots (fx, fy, fz, tx, ty, tz):

- **Gray solid line** — full actual time series of the target cohort (all 11 points including hidden)
- **Colored dashed line** — LSTM prediction extending from the last observed point to the predicted test points
- **Colored circles** — LSTM predicted values at test time points
- **Black triangles (▲)** — actual hidden ground-truth values

## Usage

```bash
python lstm_forecast_v2.py
```

Output is saved to `/home/biohpc/Dental/lstm/results/`. To run in the background on BioHPC:

```bash
nohup python3 lstm_forecast_v2.py > forecast_v2.log 2>&1 &
```

Already-completed plots are automatically skipped on re-run.

## Output

- `results/lstm_forecast_v2_{sheet}_cohort{N}_k{k}.png` — 60 PNGs total (5 cohorts × 3 k-values × 4 sheets)
- MAE summary printed to stdout at the end of the run

---

## Results

### DPA 0.25mm — U6

**k=1** (predict last 1 time point: 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA025_U6_cohort1_k1.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort2_k1.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort3_k1.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort4_k1.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort5_k1.png) |

**k=2** (predict last 2 time points: 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA025_U6_cohort1_k2.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort2_k2.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort3_k2.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort4_k2.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort5_k2.png) |

**k=3** (predict last 3 time points: 6 Days, 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA025_U6_cohort1_k3.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort2_k3.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort3_k3.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort4_k3.png) | ![](results/lstm_forecast_v2_DPA025_U6_cohort5_k3.png) |

---

### DPA 0.25mm — U7

**k=1** (predict last 1 time point: 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA025_U7_cohort1_k1.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort2_k1.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort3_k1.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort4_k1.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort5_k1.png) |

**k=2** (predict last 2 time points: 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA025_U7_cohort1_k2.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort2_k2.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort3_k2.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort4_k2.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort5_k2.png) |

**k=3** (predict last 3 time points: 6 Days, 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA025_U7_cohort1_k3.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort2_k3.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort3_k3.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort4_k3.png) | ![](results/lstm_forecast_v2_DPA025_U7_cohort5_k3.png) |

---

### DPA 0.5mm — U6

**k=1** (predict last 1 time point: 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA05_U6_cohort1_k1.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort2_k1.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort3_k1.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort4_k1.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort5_k1.png) |

**k=2** (predict last 2 time points: 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA05_U6_cohort1_k2.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort2_k2.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort3_k2.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort4_k2.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort5_k2.png) |

**k=3** (predict last 3 time points: 6 Days, 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA05_U6_cohort1_k3.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort2_k3.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort3_k3.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort4_k3.png) | ![](results/lstm_forecast_v2_DPA05_U6_cohort5_k3.png) |

---

### DPA 0.5mm — U7

**k=1** (predict last 1 time point: 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA05_U7_cohort1_k1.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort2_k1.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort3_k1.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort4_k1.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort5_k1.png) |

**k=2** (predict last 2 time points: 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA05_U7_cohort1_k2.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort2_k2.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort3_k2.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort4_k2.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort5_k2.png) |

**k=3** (predict last 3 time points: 6 Days, 7 Days, 14 days)

| Cohort 1 | Cohort 2 | Cohort 3 | Cohort 4 | Cohort 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![](results/lstm_forecast_v2_DPA05_U7_cohort1_k3.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort2_k3.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort3_k3.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort4_k3.png) | ![](results/lstm_forecast_v2_DPA05_U7_cohort5_k3.png) |
