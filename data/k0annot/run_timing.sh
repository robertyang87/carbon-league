#!/bin/bash
# C1 timing slice: games 994400-994403, 62 workers, pinned numpy threads.
set -u
ROOT=/root/carbon/k0annot
PKG=$ROOT/pkg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PKG/code" || exit 1
date +%s > "$ROOT/timing_start.txt"
date "+%F %T %Z" >> "$ROOT/timing_start.txt"
/root/carbon/.venv/bin/python k0_annotate.py \
  --weights "$PKG/weights/cnet2_m3s2r1_heads_v2.npz" \
  --replays "$PKG/replays" \
  --games "$ROOT/games_timing_c1.json" \
  --out "$ROOT/out_timing" \
  --isolation-file "$PKG/iso_shard0.json" \
  --isolation-file "$PKG/iso_test323.json" \
  --workers 62 > "$ROOT/timing.log" 2>&1
echo "rc=$?" >> "$ROOT/timing.log"
date +%s > "$ROOT/timing_end.txt"
