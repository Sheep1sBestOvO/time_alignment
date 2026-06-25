#!/bin/bash
#SBATCH --job-name=align_tmpldrift
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/align_tmpldrift_%A_%a.out
#SBATCH --error=logs/align_tmpldrift_%A_%a.err

set -euo pipefail

# ---- EDIT FOR YOUR CLUSTER ----
REPO=/scratch/yz8680/generic-neuromotor-interface
source "$REPO/.venv/bin/activate"
# -------------------------------

cd "$REPO"
mkdir -p logs

USERS=(000 001 002)
U=${USERS[$SLURM_ARRAY_TASK_ID]}
HDF5="data/discrete_gestures_user_${U}_dataset_000.hdf5"
OUTROOT="outputs/template_drift_diagnostic/user_${U}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment \
  template-drift-diagnostic \
  "$HDF5" "$OUTROOT" \
  --num-prompts 2000 \
  --shift-model prompt-delay \
  --delay-mean 0.25 --delay-std 0.07 --delay-range 0.05 0.55 \
  --uncertainty -0.10 0.70 \
  --feature mpf \
  --template-estimator rerp \
  --reference-template-estimator rerp \
  --template-ridge 0.001 \
  --pre 0.3 --post 0.9 \
  --iterations 5 --beam-width 30 --candidate-step 0.02 \
  --min-event-separation 0.02 \
  --prompt-prior-weight 0.0 \
  --max-template-shift 0.5 \
  --oracle-init \
  --seed 0
