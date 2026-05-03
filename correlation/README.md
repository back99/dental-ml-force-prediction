# Correlation Analysis

Computes the pairwise Pearson correlation between the 6 orthodontic force/moment components on the combined DPA 0.25mm + 0.5mm dataset, and renders it as a heatmap.

## Forces & Moments

- **Forces:** Fx, Fy, Fz (N)
- **Moments:** Tx, Ty, Tz (Nmm)

## Inputs

- `dpa_025.csv`, `dpa_05.csv` — produced by `../xgboost/extract_dpa.py` from `smith_dataset.csv`. Run that script first.

## Scripts

| File | Description |
|---|---|
| `correlation.py` | Loads both CSVs, concatenates them, computes the correlation matrix, prints it, and saves a `seaborn` heatmap (`coolwarm`, annotated, centered at 0). |

## How to run

```bash
cd correlation
python correlation.py
# → prints the 6x6 correlation matrix
# → writes correlation_matrix.png
```

## Results

### Correlation Matrix

![Correlation matrix between Fx/Fy/Fz/Tx/Ty/Tz on combined 0.25 + 0.5mm DPA data](results/correlation_matrix.png)
