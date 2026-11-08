#!/bin/bash
# v17 decision needs the same-tool v16b planter baseline -> it moves right after the three v17 exams
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
kill -9 $(pgrep -f "bash /root/q2.sh") 2>/dev/null
B=../.venv/bin/python
while pgrep -f "bigeval.py exp/submission_cnet2_v17c_nc.py exp/cnet2_wsstand" >/dev/null; do sleep 15; done
echo "$(date +%T) v17c ws" >> ../data/q2.progress
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/submission_cnet2_v17c_nc.py exp/$f.py $s 100 c_v17c_$n >> ../data/v17c.log 2>&1; echo "$(date +%T) v17c $n" >> ../data/q2.progress
done
$B bigeval.py exp/submission_cnet2_v16s_mixnc.py exp/cnet2_m2s2stand.py 996000 200 c_v16b_m2s2_pl >> ../data/plbase.log 2>&1; echo "$(date +%T) v16b_pl" >> ../data/q2.progress
for v in v18a v18b; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/cnet2_m2s2stand.py 996000 200 c_${v}_m2s2 >> ../data/v18.log 2>&1; echo "$(date +%T) $v" >> ../data/q2.progress; done
echo DONE > ../data/q2.done
