#!/usr/bin/env bash
# Run from the repository root. Training and holdout here use only our frozen it45 opponent.
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u reports/sapling_1200k/terminal_blend/train.py
python reports/sapling_1200k/terminal_blend/attribution.py
python reports/sapling_1200k/terminal_blend/economy.py
