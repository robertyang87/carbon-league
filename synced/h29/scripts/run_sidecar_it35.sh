#!/bin/bash
# ckpt_35 sidecar (Echo4 + Echo2 fields): 4 combos x 4 chunks of 25 seeds = 16 workers.
# e4cand35/e4base35: field m35_e4 / m35base_e4 (first 100 seeds 20294101-20294200)
# e2cand35/e2base35: field ppou2_m_it35_echo2 / ppou2_base_e2_a (first 100 seeds 982000-982099)
cd /root/carbon/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
SC=/root/carbon/remote_sidecar_trees.py
LIVE=/root/carbon/data/bigeval_live
OUT=/root/carbon/sidecar_it35
mkdir -p "$OUT"
run_chunk() { # combo a b field start idx
  $PY $SC "$2" "$3" "$LIVE/$4.jsonl" 25 "$OUT/sidecar_it35_$1_c$6.jsonl" "$5" \
    > "$OUT/sidecar_it35_$1_c$6.log" 2>&1
}
for combo in "e4cand35 exp/ppou2_m_it35.py packs/echo4_int8.py m35_e4" \
             "e4base35 exp/m3s2r1_base.py packs/echo4_int8.py m35base_e4" \
             "e2cand35 exp/ppou2_m_it35.py exp/echo2.py ppou2_m_it35_echo2" \
             "e2base35 exp/m3s2r1_base.py exp/echo2.py ppou2_base_e2_a"; do
  set -- $combo
  for c in 0 1 2 3; do
    start=$((c * 25))
    (run_chunk "$1" "$2" "$3" "$4" "$start" "$c") &
  done
done
wait
echo ALL_CHUNKS_DONE
grep -h sampled $OUT/sidecar_it35_*_c*.log
