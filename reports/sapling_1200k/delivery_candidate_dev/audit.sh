#!/usr/bin/env bash
# Reconstruct the completed four-game block. Candidate and parent always see the same public history.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
base=reports/sapling_1200k
run="$base/delivery_candidate_dev"
actors=data/weights/sapling_1200k
python "$base/official_game_replay.py" "$run" --out "$run/official_replay.json"
python "$base/delivery_action_replay.py" "$run" "$actors/freight_it50_v3.py" "$actors/hybrid_it50_v3.py" \
   "$run/action_attribution.json.gz" --manifest "$base/freight_identity.json" --parent-sha256 b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca
