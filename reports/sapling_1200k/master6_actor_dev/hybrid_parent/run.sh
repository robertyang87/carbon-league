#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
r=reports/sapling_1200k
out="$r/master6_actor_dev/hybrid_parent"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u "$r/tree_pack_arena.py" --maps 621458920 --cpus 1 2 --weights "$out/build_manifest.json" \
  --pack data/weights/sapling_1200k/hybrid_it50_v3.py --pack-sha256 b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca --run "$out"
if [[ ! -f "$out/official_replay.json" ]]; then python "$r/official_game_replay.py" "$out" --out "$out/official_replay.json"; fi
