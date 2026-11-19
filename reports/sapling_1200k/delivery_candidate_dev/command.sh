#!/usr/bin/env bash
# Reproduce the frozen development block from the repository root.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --pack data/weights/sapling_1200k/freight_it50_v3.py \
    --pack-sha256 d8778add6bebb8c9c2f88b9171ac706b801f4f88c8e274b06b23a2a606862a18 \
    --weights reports/sapling_1200k/freight_identity.json \
    --maps 620224310 620224311 --cpus 2 3 \
    --run reports/sapling_1200k/delivery_candidate_dev
