#!/usr/bin/env bash
# Independent development block, same maps/seats as the archived hybrid parent.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/d1_sapling_arena.py \
  --pack data/weights/sapling_1200k/d1_sapling.py \
  --weights reports/sapling_1200k/d1_sapling_dev/build_manifest.json \
  --pack-sha256 429ec9a2bb84f77485b4f8830e71b252267705ffd8ce854a1cc35a03c446a367 \
  --maps 620224310 620224311 --cpus 0 1 --run reports/sapling_1200k/d1_sapling_dev
