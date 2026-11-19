#!/usr/bin/env bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u reports/sapling_1200k/tree_pack_arena.py \
  --weights data/weights/sapling_1200k/soft_it50_v5.npz \
  --pack data/weights/sapling_1200k/soft_it50_v5_ambush5.py \
  --pack-sha256 5e03ebaf14b243570b7fb381c9ae6b8ee8984c453e6fa59868a401aa149e94b0 \
  --run reports/sapling_1200k/ambush5_dev --maps 620106811 --cpus 2 3
python reports/sapling_1200k/ambush5_replay.py \
  --run reports/sapling_1200k/ambush5_dev --out reports/sapling_1200k/ambush5_live.json
