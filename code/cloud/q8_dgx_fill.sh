#!/bin/bash
# DGX CPU filler: remaining q8 chunks (1C-v9 second map set, 50-game chunks) from the last chunk backwards, so host 4
# (which runs them forwards) meets finished files and skips them (sync_hosts pushes complete files only).
# Yields to the pure-net exams (v9econ): its bigeval processes are SIGSTOPped while any v9econ game runs. _nc packs have
# no clock caps, so pausing does not change any game.
cd /home/arm/junyang/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=12
B=../.venv/bin/python; P=../data/q8_dgx.progress
( while true; do
    if pgrep -f "[b]igeval.py exp/cnet2_" >/dev/null; then pkill -STOP -f "[b]igeval.py exp/submission_cnet2_v17"; else pkill -CONT -f "[b]igeval.py exp/submission_cnet2_v17"; fi
    sleep 10; pgrep -f "[q]8_dgx_fill.sh" >/dev/null || exit; done ) &
run() { local f=../data/bigeval_live/$5.jsonl
  if [ -f "$f" ] && [ "$(grep -c '"margin": -\?[0-9]' "$f")" -ge "$4" ]; then echo "$(date +%T) skip $5" >> $P; return; fi
  rm -f "$f"; $B bigeval.py exp/$1.py exp/$2.py $3 $4 $5 >> ../data/q8_dgx.log 2>&1; echo "$(date +%T) $5 $(wc -l < $f)" >> $P; }
for c in 3 2; do s=$((997200 + 50 * c))
  run submission_cnet2_v17d2_nc opp_1cz_v9 $s 50 c_v17d2_v9b_p$c
  run submission_cnet2_v17c_nc opp_1cz_v9 $s 50 c_v17c_v9b_p$c
done
echo DONE >> $P
