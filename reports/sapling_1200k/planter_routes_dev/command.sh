#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
work=reports/sapling_1200k/planter_routes_dev
taskset -c 0,3 /workspace/carbon-review-venv/bin/python -u reports/sapling_1200k/tree_pack_arena.py --maps 621456920 --cpus 0 3 --weights "$work/build_manifest.json" --pack data/weights/sapling_1200k/planter_routes_it50_v3.py --pack-sha256 311631ce99d160081aa4f97d1a2801c199d830b22b02b20d7c80b233a36bbf27 --run "$work/candidate"
