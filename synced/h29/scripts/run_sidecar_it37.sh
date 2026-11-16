#!/bin/bash
# ckpt_37 sidecar (Echo4 field only, per PPOU2_AMENDMENT_THREE §6):
# 2 combos x 4 chunks of 25 seeds = 8 workers.
# e4cand37/e4base37: field m37_e4 / m37base_e4 (first 100 seeds 20296101-20296200)
# Echo2 sidecar NOT run (not computed; amendment says do not backfill).
cd /root/carbon/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
SC=/root/carbon/remote_sidecar_trees.py
LIVE=/root/carbon/data/bigeval_live
OUT=/root/carbon/sidecar_it37
mkdir -p "$OUT"
run_chunk() { # combo a b field start idx
  $PY $SC "$2" "$3" "$LIVE/$4.jsonl" 25 "$OUT/sidecar_it37_$1_c$6.jsonl" "$5" \
    > "$OUT/sidecar_it37_$1_c$6.log" 2>&1
}
for combo in "e4cand37 exp/ppou2_m_it37.py packs/echo4_int8.py m37_e4" \
             "e4base37 exp/m3s2r1_base.py packs/echo4_int8.py m37base_e4"; do
  set -- $combo
  for c in 0 1 2 3; do
    start=$((c * 25))
    (run_chunk "$1" "$2" "$3" "$4" "$start" "$c") &
  done
done
wait
echo ALL_CHUNKS_DONE
grep -h sampled $OUT/sidecar_it37_*_c*.log
