#!/bin/bash
# fixed-compute proxies for the platform budget (time caps are not portable to the oversubscribed cloud box)
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
B=../.venv/bin/python
for r in 4 6; do
  rm -f ../data/rs_stats_$r.jsonl
  CS_STATS=$(pwd)/../data/rs_stats_$r.jsonl $B bigeval.py exp/cnet2_v7s_mfxn_r$r.py exp/submission_cnet2_v13.py 992000 200 c_v7s_r${r}_v13 > /dev/null 2>&1
done
echo DONE > ../data/rs.done
