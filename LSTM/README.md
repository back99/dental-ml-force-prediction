# LSTM Cohort Prediction

LSTM (Long Short-Term Memory) models for predicting dental aligner force/moment patterns across patient cohorts.

## Files

| File | Description |
|------|-------------|
| `lstm_cohort.py` | Single run: train on Cohorts 1-4, predict Cohort 5 |
| `lstm_loo.py` | Leave-One-Out CV: for each cohort i, train on other 4, predict cohort i |

## Data

- `data/cohort_1.xlsx` ~ `cohort_5.xlsx`
- 4 sheets per file: `DPA025_U6`, `DPA025_U7`, `DPA05_U6`, `DPA05_U7`
- Columns: `time` (string labels: "0h", "8h", ..., "14 days"), `fx`, `fy`, `fz`, `tx`, `ty`, `tz`

## Model

```
Input: normalized time (scalar) → LSTM(128 hidden, 2 layers) → Dense(64) → output (scalar)
```

- Hidden size: 128, Layers: 2, Dropout: 0.2
- Epochs: 300, LR: 1e-3 (StepLR ×0.5 every 100 epochs), Batch: 512
- Device: CPU

## Usage

```bash
# Single run (Cohorts 1-4 → predict 5)
python lstm_cohort.py

# Leave-One-Out CV (5 folds)
python lstm_loo.py
```

## Output

- `results/lstm_{sheet}.png` — single run prediction plots (4 sheets × 6 components)
- `results/loo/lstm_loo_{sheet}_predict_cohort{N}.png` — LOO plots (4 sheets × 5 cohorts = 20 plots)
- MAE summary printed to stdout

## Why LOO-CV?

With only 5 cohorts, a single train/test split is too sensitive to which cohort is held out. LOO-CV trains 5 separate models (each leaving out one cohort) and averages MAE across all folds, giving a more reliable estimate of generalization performance.
