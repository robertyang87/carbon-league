#!/bin/bash
# C2 timing slice: games 994596-994599, 62 workers, threads pinned 1.
cd /root/carbon/k0pkg/code || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
META=/root/carbon/data/k0_annot_c2_timing.meta
echo "START_EPOCH $(date +%s)" > "$META"
echo "START_ISO $(date -Is)" >> "$META"
setsid /root/carbon/.venv/bin/python k0_annotate.py \
  --weights /root/carbon/k0pkg/weights/cnet2_m3s2r1_heads_v2.npz \
  --replays /root/carbon/k0pkg/replays \
  --games /root/carbon/k0pkg/games_c2_timing.json \
  --out /root/carbon/data/k0_annot_c2 \
  --isolation-file /root/carbon/k0pkg/iso_shard0.json \
  --isolation-file /root/carbon/k0pkg/iso_test323.json \
  --workers 62 > /root/carbon/data/k0_annot_c2_timing.log 2>&1 < /dev/null &
PID=$!
echo "MAIN_PID $PID" >> "$META"
echo "PGID $PID" >> "$META"
echo "WORKERS 62" >> "$META"
echo "launched pid=$PID"
