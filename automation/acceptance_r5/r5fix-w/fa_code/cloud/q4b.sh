#!/bin/bash
# q4 without the secondary endgame-rule exams vs v13 / WB-532 (cut 2026-11-09 on request). Waits for the running dagger4c.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=60
B=../.venv/bin/python; P=../data/q4.progress
step() { echo "$(date +%T) $1" >> $P; }
while pgrep -f "[d]agger_gen.py ../data/dagger4c" >/dev/null; do sleep 15; done; step "dagger4c"
$B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17d2_m2s2 >> ../data/q4.log 2>&1; step "v17d2 m2s2"
$B bigeval.py exp/submission_cnet2_v17c_nc.py exp/opp_1cz_v9.py 997000 200 c_v17c_v9 >> ../data/q4.log 2>&1; step "v17c v9"
$B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/opp_1cz_v9.py 997000 200 c_v17d2_v9 >> ../data/q4.log 2>&1; step "v17d2 v9"
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/$f.py $s 100 c_v17d2_$n >> ../data/q4.log 2>&1; step "v17d2 $n"; done
for v in v17t v17te v17tb; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/opp_1cz_v9.py 997000 200 c_${v}_v9 >> ../data/q4.log 2>&1; step "$v v9"; done
$B bigeval.py exp/submission_cnet2_v17te_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17te_m2s2 >> ../data/q4.log 2>&1; step "v17te m2s2"
echo DONE > ../data/q4.done
