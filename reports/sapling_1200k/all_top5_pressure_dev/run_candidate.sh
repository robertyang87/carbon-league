#!/usr/bin/env bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
base=reports/sapling_1200k
run="$base/all_top5_pressure_dev"
python -u "$base/tree_pack_arena.py" --maps 621455920 --cpus 0 3 \
    --weights "$run/build_manifest.json" --pack data/weights/sapling_1200k/pressure_return_it50_v3.py \
    --pack-sha256 21c56a4cd86ea136f089da07a8a6a456b936e18c8283769bca8943fac2faaf90 \
    --run "$run/candidate"
