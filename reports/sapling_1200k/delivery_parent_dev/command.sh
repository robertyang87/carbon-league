#!/usr/bin/env bash
# Frozen map/seat block for the unchanged hybrid. Run from the repository root.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --weights reports/sapling_1200k/delivery_parent_dev/build_manifest.json \
    --pack data/weights/sapling_1200k/hybrid_it50_v3.py \
    --pack-sha256 b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca \
    --maps 620224310 620224311 --cpus 0 1 \
    --run reports/sapling_1200k/delivery_parent_dev
