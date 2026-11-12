#!/bin/bash
# q6b = q6 reordered after v18e (-130 +- 20 vs v17d2): gate exams and fresh-map stand-in first, go-home-rule variants last.
# q6 = q5 reordered after v18c (threat-first cost the hunting slot): v18e first, threat-first runs last.
# Waits for any running bigeval (the queue it replaces) before touching result files.
# Resumable exam queue (any host): an exam whose result file already holds N games is skipped (results are pushed
# from the Mac by bringup.sh), a partial one is rerun whole. Same exams and seeds as q4b.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=$(( $(nproc) - 4 ))
B=../.venv/bin/python; P=../data/q4.progress
run() {  # run PACK OPP SEED N TAG
  local f=../data/bigeval_live/$5.jsonl
  if [ -f "$f" ] && [ "$(grep -c '"margin": -\?[0-9]' "$f")" -ge "$4" ]; then echo "$(date +%T) skip $5 (have $4)" >> $P; return; fi
  rm -f "$f"; $B bigeval.py exp/$1.py exp/$2.py $3 $4 $5 >> ../data/q4.log 2>&1; echo "$(date +%T) $5" >> $P
}
while pgrep -f "[b]igeval.py exp" >/dev/null; do sleep 15; done
run submission_cnet2_v17d2_nc submission_cnet2_v13 992000 100 c_v17d2_v13
run submission_cnet2_v17d2_nc opp_wb532 995000 100 c_v17d2_wb532
run submission_cnet2_v17d2_nc cnet2_m2s2stand 996200 200 c_v17d2_m2s2b
run submission_cnet2_v17c_nc cnet2_m2s2stand 996200 200 c_v17c_m2s2b
run submission_cnet2_v17d2_nc cnet2_wsstand 999500 100 c_v17d2_ws
run submission_cnet2_v18e_nc opp_1cz_v9 997000 200 c_v18e_v9
for v in v17t v17te; do run submission_cnet2_${v}_nc opp_1cz_v9 997000 200 c_${v}_v9; done
run submission_cnet2_v18c_nc opp_1cz_v9 997000 200 c_v18c_v9
run submission_cnet2_v17tb_nc opp_1cz_v9 997000 200 c_v17tb_v9
echo DONE > ../data/q4.done
