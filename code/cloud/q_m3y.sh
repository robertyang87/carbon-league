#!/bin/bash
# q_m3y.sh G W... : wait for each weight file, then pure-net exam vs 3 training opps x 200 (seeds 994000), 10 workers each
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
G=$1; shift
for W in "$@"; do until [ -f ../data/weights/$W.npz.ok ]; do sleep 20; done; done
for W in "$@"; do for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 ${G}_${W}_$O > ../data/${G}_${W}_$O.out 2>&1 < /dev/null &
done; done
wait; echo M3Y_DONE > ../data/${G}.done
