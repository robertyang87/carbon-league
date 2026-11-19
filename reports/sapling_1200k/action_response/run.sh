#!/usr/bin/env bash
# Run from repo root with a verified old training replay directory as argument.
# SAPLING_RESPONSE_CPU optionally selects one available CPU on another host.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>reports/sapling_1200k/action_response/.run.lock
flock -n 9
base=reports/sapling_1200k
python "$base/action_response.py" --replays "$1" --stage trainval
python "$base/action_response_fit.py" fit
python "$base/action_response.py" --replays "$1" --stage test
python "$base/action_response_fit.py" evaluate
