#!/bin/bash
# re-prioritised queue: v17 decision first (Master2s 200, other three exams 100 each), then v18, then baselines
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
pkill -f "/root/plbase.sh"; pkill -f "/root/v18.sh"
until [ "$(wc -l < ../data/bigeval_live/c_v17c_m2s2.jsonl)" -ge 200 ]; do sleep 15; done
pkill -f "/root/v17c.sh"; sleep 2; pkill -f "bigeval.py exp/submission_cnet2_v17c_nc.py exp/cnet2_wsstand"; sleep 5
ps -eo pid,ppid,args | grep "[m]ultiprocessing\|[c]oncurrent" | awk '$2==1 {print $1}' | xargs -r kill 2>/dev/null
rm -f ../data/bigeval_live/c_v17c_ws.jsonl
B=../.venv/bin/python
for o in ws:cnet2_wsstand:999500 v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/submission_cnet2_v17c_nc.py exp/$f.py $s 100 c_v17c_$n >> ../data/v17c.log 2>&1; echo "$(date +%T) v17c $n" >> ../data/q2.progress
done
for v in v18a v18b; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/cnet2_m2s2stand.py 996000 200 c_${v}_m2s2 >> ../data/v18.log 2>&1; echo "$(date +%T) $v" >> ../data/q2.progress; done
$B bigeval.py exp/submission_cnet2_v16s_mixnc.py exp/cnet2_m2s2stand.py 996000 200 c_v16b_m2s2_pl >> ../data/plbase.log 2>&1; echo "$(date +%T) v16b_pl" >> ../data/q2.progress
echo DONE > ../data/q2.done
