#!/bin/bash
#SBATCH --job-name=align_oracleinit
#SBATCH --array=0-2                     # one task per user (000, 001, 002)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/align_oracleinit_%A_%a.out
#SBATCH --error=logs/align_oracleinit_%A_%a.err
# No GPU needed: the MPF featurizer runs on CPU (torch, no .cuda calls).

set -euo pipefail

# ------------------------------------------------------------------
# EDIT THESE FOR YOUR CLUSTER (NYU Greene etc.)
# ------------------------------------------------------------------
#SBATCH_ACCOUNT / #SBATCH --partition:  add above if your cluster requires them, e.g.
#   #SBATCH --account=pr_XXX
#   #SBATCH --partition=cpu
REPO=/scratch/yz8680/generic-neuromotor-interface     # <-- repo path ON THE CLUSTER

# Environment: build this on the cluster (a mac .venv will NOT run on Linux).
# Option A — venv you created on the cluster:
#   source "$REPO/.venv/bin/activate"
# Option B — conda:
#   module purge
#   eval "$(conda shell.bash hook)"
#   conda activate neuromotor
source "$REPO/.venv/bin/activate"                     # <-- adjust to your env
# ------------------------------------------------------------------

cd "$REPO"
mkdir -p logs

USERS=(000 001 002)
U=${USERS[$SLURM_ARRAY_TASK_ID]}
HDF5="data/discrete_gestures_user_${U}_dataset_000.hdf5"
OUTROOT="outputs/oracle_init_eval/user_${U}"

# Use OMP/MKL threads = allocated cpus so torch/numpy don't oversubscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

CMD="python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval"

# Shared, paper-style settings. FIXED window (no --auto-windows / no adaptive),
# MPF + rERP, paper-style EM (recenter off, which is the default here).
COMMON="\
  --num-prompts 2000 \
  --shift-model prompt-delay \
  --delay-mean 0.25 --delay-std 0.07 --delay-range 0.05 0.55 \
  --uncertainty -0.10 0.70 \
  --feature mpf --template-estimator rerp --template-ridge 0.001 \
  --pre 0.3 --post 0.9 \
  --iterations 5 --beam-width 30 --candidate-step 0.02 \
  --min-event-separation 0.02 \
  --seed 0 \
  --with-oracle"

echo "=========================================================="
echo "User ${U}: BASELINE (cold-start EM)"
echo "=========================================================="
$CMD "$HDF5" "${OUTROOT}/baseline" $COMMON

echo "=========================================================="
echo "User ${U}: ORACLE-INIT (seed EM iter 1 with ground-truth templates)"
echo "=========================================================="
$CMD "$HDF5" "${OUTROOT}/oracle_init" $COMMON --oracle-init

echo "DONE user ${U}"
