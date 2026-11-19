#!/usr/bin/env bash
# Run from the repository root. The evidence directory resumes existing checked results.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --weights data/weights/sapling_1200k/soft_it50_v5.npz \
    --pack data/weights/sapling_1200k/soft_it50_v5_agent.py \
    --pack-sha256 dceed7891e9ff25cf9e211e1c9b61d979d27a253b4c0935f2439804372b31389 \
    --maps 620106811 620106812 --cpus 0 1 \
    --run reports/sapling_1200k/tree_exam_soft_it50_v5
