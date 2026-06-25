#!/bin/bash
#SBATCH --job-name=align_warmstart
#SBATCH --array=0-2                     # one task per user (000, 001, 002)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/align_warmstart_%A_%a.out
#SBATCH --error=logs/align_warmstart_%A_%a.err
# CPU-only (MPF featurizer runs on CPU torch).

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
OUTROOT="outputs/warmstart_eval/user_${U}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

CMD="python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval"

# Same fixed window / MPF+rERP / shift model as the oracle-init run, so numbers
# are directly comparable. All conditions are UNSUPERVISED (no ground truth).
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

echo "### user ${U}: recenter only ###"
$CMD "$HDF5" "${OUTROOT}/recenter"        $COMMON --recenter

echo "### user ${U}: coarse-to-fine only ###"
$CMD "$HDF5" "${OUTROOT}/coarse2fine"     $COMMON --coarse-to-fine 4

echo "### user ${U}: recenter + coarse-to-fine ###"
$CMD "$HDF5" "${OUTROOT}/recenter_c2f"    $COMMON --recenter --coarse-to-fine 4

echo "DONE user ${U}"
