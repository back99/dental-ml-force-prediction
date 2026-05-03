#!/bin/bash
# ============================================================================
# SLURM batch script — GPR force prediction on Libra HPC
# ============================================================================
# Submits the GPR pipeline (Step 1 → Step 2) to a single L40S GPU node.
#
# Usage:
#   cd dental-ml-force-prediction/gpr
#   sbatch run_gpr.sbatch              # runs both steps
#   sbatch run_gpr.sbatch step1        # runs only gpr_prediction.py
#   sbatch run_gpr.sbatch step2        # runs only gpr_with_sim.py
#
# Prereqs:
#   - dpa_025.csv, dpa_05.csv exist in this directory
#     (run ../xgboost/extract_dpa.py first)
#   - sim_075_*.csv exist (run ../xgboost/generate_sim_075.py and copy here,
#     or symlink) — required for Step 2
#   - Conda env "dental-gpr" with torch + gpytorch + xgboost installed
# ============================================================================

#SBATCH --job-name=dental-gpr
#SBATCH --partition=gpu                  # TODO: replace with your Libra partition (e.g. gpu, gpu-l40s)
#SBATCH --account=<YOUR_ACCOUNT>         # TODO: replace with your SLURM account
#SBATCH --gres=gpu:l40s:1                # 1× NVIDIA L40S; use gpu:1 if generic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00                  # 2h is generous; tune to your run time
#SBATCH --output=logs/gpr_%j.out
#SBATCH --error=logs/gpr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sjung15@slu.edu

set -euo pipefail
mkdir -p logs

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "GPU(s):        $CUDA_VISIBLE_DEVICES"
echo "Working dir:   $(pwd)"
echo "Started:       $(date)"
echo "========================================"

# ----- Environment -----
# Adjust module names / versions to whatever Libra exposes (`module avail` on a login node)
module purge
module load cuda/12.1
module load miniconda3            # or: module load anaconda3

# Activate Conda env (create once with: conda env create -f environment.yml)
source activate dental-gpr        # rename if your env differs

# Sanity check — should print the L40S device
python - <<'PY'
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

# ----- Pipeline -----
STEP="${1:-all}"

case "$STEP" in
    step1)
        echo "[$(date)] Running Step 1: gpr_prediction.py"
        python gpr_prediction.py
        ;;
    step2)
        echo "[$(date)] Running Step 2: gpr_with_sim.py"
        python gpr_with_sim.py
        ;;
    all)
        echo "[$(date)] Running Step 1: gpr_prediction.py"
        python gpr_prediction.py
        echo "[$(date)] Running Step 2: gpr_with_sim.py"
        python gpr_with_sim.py
        ;;
    *)
        echo "Unknown step: $STEP (use step1 | step2 | all)" >&2
        exit 2
        ;;
esac

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
