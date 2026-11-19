#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
r=reports/sapling_1200k
out="$r/all_top5_pressure_dev/parent"
python -u "$r/tree_pack_arena.py" --pack data/weights/sapling_1200k/hybrid_it50_v3.py \
  --pack-sha256 b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca \
  --weights "$out/build_manifest.json" --maps 621455920 --cpus 1 2 --run "$out"
if [[ ! -f "$out/official_replay.json" ]]; then
  python "$r/official_game_replay.py" "$out" --out "$out/official_replay.json"
fi
