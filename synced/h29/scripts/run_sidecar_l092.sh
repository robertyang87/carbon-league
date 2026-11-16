#!/bin/bash
# l092 it=20 sidecar: 2 fields (Master2 + Echo4) x 2 combos x 4 chunks of 25 = 16 workers.
# m2cand/m2base: field l092_it20_m2 / frozen ppou2_base_m2_a (seeds 980000-980099)
# e4cand/e4base: field l092_it20_e4 / l092base_e4 (seeds 20300101-20300200)
cd /root/carbon/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
SC=/root/carbon/remote_sidecar_trees.py
LIVE=/root/carbon/data/bigeval_live
OUT=/root/carbon/sidecar_l092
mkdir -p "$OUT"
run_chunk() { # combo a b field start idx
  $PY $SC "$2" "$3" "$LIVE/$4.jsonl" 25 "$OUT/sidecar_l092_$1_c$6.jsonl" "$5" \
    > "$OUT/sidecar_l092_$1_c$6.log" 2>&1
}
for combo in "m2cand exp/ppou2_l092_it20.py exp/opp_master2r.py l092_it20_m2" \
             "m2base exp/m3s2r1_base.py exp/opp_master2r.py ppou2_base_m2_a" \
             "e4cand exp/ppou2_l092_it20.py packs/echo4_int8.py l092_it20_e4" \
             "e4base exp/m3s2r1_base.py packs/echo4_int8.py l092base_e4"; do
  set -- $combo
  for c in 0 1 2 3; do
    start=$((c * 25))
    (run_chunk "$1" "$2" "$3" "$4" "$start" "$c") &
  done
done
wait
echo ALL_CHUNKS_DONE
grep -h sampled $OUT/sidecar_l092_*_c*.log
