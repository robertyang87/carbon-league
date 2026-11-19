#!/usr/bin/env bash
# Run from the repository root after all four action tapes are present.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
base=reports/sapling_1200k
run="$base/delivery_parent_dev"
python "$base/official_game_replay.py" "$run" --out "$run/official_replay.json"
python "$base/turning_point_ledger.py" "$run" --out "$run/lead_ledger"
python "$run/attribution_check.py" > "$run/attribution.json.tmp"
mv "$run/attribution.json.tmp" "$run/attribution.json"
