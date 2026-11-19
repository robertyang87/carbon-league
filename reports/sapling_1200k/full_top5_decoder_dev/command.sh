#!/usr/bin/env bash
# Run from the repository root. This map is reserved for development comparisons on known opponents.
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CNET2_QMODE=int8
dir=reports/sapling_1200k/full_top5_decoder_dev
python "$dir/build.py"
python -u reports/sapling_1200k/tree_pack_arena.py \
    --pack data/weights/sapling_1200k/full_top5_decoder_agent.py \
    --pack-sha256 a957095b5aac296d4ff3487d2e12597893af8c13a009509954618368f2d534b4 \
    --weights data/weights/sapling_1200k/full_top5_decoder.npz \
    --maps 620106811 --cpus 2 3 --run "$dir"
python reports/sapling_1200k/official_game_replay.py "$dir" --out "$dir/official_replay.json"
python "$dir/compare.py"
