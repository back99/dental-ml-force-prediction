# Correlation Analysis

Analyzes the correlation between 6 orthodontic forces and moments across different aligner thicknesses.

## Forces & Moments
- **Forces:** Fx, Fy, Fz (N)
- **Moments:** Tx, Ty, Tz (Nmm)

## Data
- `dpa_025.csv` — 0.25mm aligner thickness data
- `dpa_05.csv` — 0.5mm aligner thickness data

## Scripts
- `correlation.py` — Computes and visualizes correlation matrix between all 6 force/moment components

## Results
| File | Description |
|------|-------------|
| `results/correlation_matrix.png` | Heatmap of pairwise correlations between Fx, Fy, Fz, Tx, Ty, Tz |
