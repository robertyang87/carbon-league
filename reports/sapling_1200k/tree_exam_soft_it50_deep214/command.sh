#!/usr/bin/env bash
# Run from the repository root. Same development maps, seats and CPU placement as 1.2M bare V5.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --weights data/weights/sapling_1200k/soft_it50_deep214.npz \
    --pack data/weights/sapling_1200k/soft_it50_deep214_agent.py \
    --pack-sha256 ef824c07042922d7d7a7115516451291e6db4c25b3ff63219fdadfc53da44a9b \
    --maps 620106811 620106812 --cpus 0 1 \
    --run reports/sapling_1200k/tree_exam_soft_it50_deep214
