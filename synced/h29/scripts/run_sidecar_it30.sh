#!/bin/bash
# ckpt_30 sidecar (Echo4 field): 2 combos x 4 chunks of 25 seeds = 8 workers.
# Combos: e4cand (ppou2_m_it30 vs Echo4), e4base (m3s2r1_base vs Echo4).
# Sample: first 100 seeds ascending of each field, same seats as field rows.
cd /root/carbon/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
SC=/root/carbon/remote_sidecar_trees.py
LIVE=/root/carbon/data/bigeval_live
OUT=/root/carbon/sidecar_it30
mkdir -p "$OUT"
run_chunk() { # combo a b field start idx
  $PY $SC "$2" "$3" "$LIVE/$4.jsonl" 25 "$OUT/sidecar_it30_$1_c$6.jsonl" "$5" \
    > "$OUT/sidecar_it30_$1_c$6.log" 2>&1
}
export -f run_chunk 2>/dev/null
for combo in "e4cand exp/ppou2_m_it30.py packs/echo4_int8.py m30_e4" \
             "e4base exp/m3s2r1_base.py packs/echo4_int8.py base_e4"; do
  set -- $combo
  for c in 0 1 2 3; do
    start=$((c * 25))
    (run_chunk "$1" "$2" "$3" "$4" "$start" "$c") &
  done
done
wait
echo ALL_CHUNKS_DONE
grep -h sampled $OUT/sidecar_it30_*_c*.log
