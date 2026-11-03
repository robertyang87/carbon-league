#!/bin/bash
set -e
export BC_NPZ=bc_data/claude_family_fixed.npz BC_WEIGHTS=shadow_weights.npz
venv/bin/python -u train_bc.py
venv/bin/python -u calibrate_base.py
venv/bin/python -u calibrate_worker.py
venv/bin/python make_shadow.py
echo ALL_DONE
