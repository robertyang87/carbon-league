#!/bin/bash
# C1 full partition run: games file passed as $1, reuses out_timing as OUT.
set -u
GAMES_FILE="${1:?games file required}"
LABEL="${2:-full}"
ROOT=/root/carbon/k0annot
PKG=$ROOT/pkg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PKG/code" || exit 1
date +%s > "$ROOT/${LABEL}_start.txt"
date "+%F %T %Z" >> "$ROOT/${LABEL}_start.txt"
/root/carbon/.venv/bin/python k0_annotate.py \
  --weights "$PKG/weights/cnet2_m3s2r1_heads_v2.npz" \
  --replays "$PKG/replays" \
  --games "$GAMES_FILE" \
  --out "$ROOT/out_timing" \
  --isolation-file "$PKG/iso_shard0.json" \
  --isolation-file "$PKG/iso_test323.json" \
  --workers 62 > "$ROOT/${LABEL}.log" 2>&1
echo "rc=$?" >> "$ROOT/${LABEL}.log"
date +%s > "$ROOT/${LABEL}_end.txt"
