#!/usr/bin/env bash
# Both seats on the preassigned development map. Exact V5 parent NPZ, interception candidate.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --weights data/weights/sapling_1200k/soft_it50_v5.npz \
    --pack data/weights/sapling_1200k/soft_it50_v5_interception.py \
    --pack-sha256 545116e662b298d37309b6d8ffd4fd434ac22ce3c133df5aecd2ddd7d5a77ea1 \
    --maps 620106811 --cpus 0 1 \
    --run reports/sapling_1200k/tree_exam_interception
