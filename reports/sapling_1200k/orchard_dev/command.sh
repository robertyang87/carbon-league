#!/usr/bin/env bash
# Frozen map/seat block for the frozen orchard candidate. Run from the repository root.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
    --weights reports/sapling_1200k/orchard_dev/build_manifest.json \
    --pack data/weights/sapling_1200k/orchard_it50_v3.py \
    --pack-sha256 1dc7345ea71ab692e3db09c00b0958f5fc086302493c5301afa89eb97125b38d \
    --maps 620224310 620224311 --cpus 0 1 \
    --run reports/sapling_1200k/orchard_dev
