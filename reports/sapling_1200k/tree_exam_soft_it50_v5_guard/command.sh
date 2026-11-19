#!/usr/bin/env bash
# Run from the repository root. Same two development maps and CPU placement as the parent.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --weights data/weights/sapling_1200k/soft_it50_v5.npz \
    --pack data/weights/sapling_1200k/soft_it50_v5_guard_agent.py \
    --pack-sha256 89df51540512d187ad64e82e21890e9f301083fc07ba06357eddaf43fc6692a2 \
    --maps 620106811 620106812 --cpus 0 1 \
    --run reports/sapling_1200k/tree_exam_soft_it50_v5_guard
