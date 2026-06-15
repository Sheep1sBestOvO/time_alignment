# Time Alignment Reproduction

This repository contains a compact, HPC-friendly copy of the discrete gesture
time-alignment reproduction code.

## What Is Included

- `generic_neuromotor_interface/discrete_gesture_alignment.py`
  - MPF-style feature extraction
  - rERP template estimation
  - beam-search time alignment
  - global template recentering
- `generic_neuromotor_interface/scripts/discrete_gesture_alignment.py`
  - CLI commands for inspection, alignment, simulated shifted-label evaluation,
    plotting, and global recentering
- `tests/test_discrete_gesture_alignment.py`
  - Unit tests for the time-alignment utilities
- `data_parts/`
  - Split HDF5 data files. Run `tools/reassemble_data.py` after cloning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then reassemble the HDF5 files:

```bash
python tools/reassemble_data.py
```

After this, the data files will be available under `data/`.

## Quick Smoke Test

```bash
python -m pytest tests/test_discrete_gesture_alignment.py
```

Run a fast shifted-label alignment check:

```bash
python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval \
  data/discrete_gestures_user_000_dataset_000.hdf5 \
  outputs/sim_shift_smoke \
  --num-prompts 24 \
  --shift-range -0.25 0.25 \
  --uncertainty -0.35 0.35 \
  --feature envelope \
  --template-estimator average \
  --iterations 3 \
  --beam-width 8 \
  --candidate-step 0.04 \
  --plot-prompts 4 \
  --seed 7 \
  --recenter
```

## Paper-Style MPF + rERP Run

```bash
python -m generic_neuromotor_interface.scripts.discrete_gesture_alignment simulate-shift-eval \
  data/discrete_gestures_user_000_dataset_000.hdf5 \
  outputs/sim_shift_user_000_mpf_rerp \
  --num-prompts 500 \
  --shift-range -0.25 0.25 \
  --uncertainty -0.35 0.35 \
  --feature mpf \
  --template-estimator rerp \
  --template-ridge 0.001 \
  --iterations 5 \
  --beam-width 30 \
  --candidate-step 0.02 \
  --min-event-separation 0.02 \
  --prompt-prior-weight 0.05 \
  --plot-prompts 8 \
  --seed 0
```

The command writes:

- `*_simulated_shifted_prompts.csv`
- `*_simulated_aligned_prompts.csv`
- `*_simulated_alignment_metrics.csv`
- `*_simulated_multichannel_alignment.svg`

## Important Note

The simulated-shift experiment does not modify raw sEMG. It treats the public
`prompts.time` as pseudo ground truth, randomly shifts those label timestamps,
and checks whether the alignment code recovers event times close to the original
public timestamps.
