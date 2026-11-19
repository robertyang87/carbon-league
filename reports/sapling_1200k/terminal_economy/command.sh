#!/usr/bin/env bash
# Run from the repository root.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/terminal_economy/run.py
python reports/sapling_1200k/terminal_economy/audit.py
