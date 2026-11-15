#!/bin/bash
# C2 K0 annotation FULL partition run: selfplay-994508..994599 (92 games).
# Same OUT as the timing run (landed shards skipped, in-flight games restart whole).
set -u
OUT=/root/carbon/data/k0_annot_c2
LOG=/root/carbon/data/k0_annot_c2_full.log
META=/root/carbon/data/k0_annot_c2_full.meta
# preserve the timing run's progress for the record
cp -f "$OUT/progress.json" /root/carbon/data/k0_annot_c2/progress_timing_run.json 2>/dev/null || true
cd /root/carbon/k0pkg/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
setsid /root/carbon/.venv/bin/python k0_annotate.py \
  --weights /root/carbon/k0pkg/weights/cnet2_m3s2r1_heads_v2.npz \
  --replays /root/carbon/k0pkg/replays \
  --games /root/carbon/k0pkg/games_c2_full.json \
  --out "$OUT" \
  --isolation-file /root/carbon/k0pkg/iso_shard0.json \
  --isolation-file /root/carbon/k0pkg/iso_test323.json \
  --workers 62 > "$LOG" 2>&1 &
PID=$!
{
  echo "START_EPOCH $(date +%s)"
  echo "START_ISO $(date -Iseconds)"
  echo "MAIN_PID $PID"
  echo "PGID $PID"
  echo "WORKERS 62"
  echo "GAMES_FILE /root/carbon/k0pkg/games_c2_full.json"
  echo "OUT $OUT"
} > "$META"
echo "LAUNCHED pid=$PID"
