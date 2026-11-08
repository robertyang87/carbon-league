#!/bin/bash
# run on the box: single-core speed + scaling (1, 32, 64 parallel pure-net games)
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64
mkdir -p ../data/cloud_logs
for w in 1 32 64; do
  t0=$(date +%s); EVO_WORKERS=$w ../.venv/bin/python bigeval.py exp/submission_cnet2_v14.py exp/submission_cnet2_v12.py 960000 $w bench_w$w >/dev/null 2>&1
  echo "workers $w games $w wall $(( $(date +%s)-t0 ))s" | tee -a ../data/cloud_logs/bench.txt
done
