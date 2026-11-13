#!/bin/bash
# runbook 4 stage 2: all 6 replicas -> Master2 real to 600 (seeds 994200 x400 added) + 3 training opps x200; style row 2 whole game
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
for W in cnet2_m3w3L cnet2_m3w3b cnet2_m3w1b cnet2_m3w1L cnet2_m3w2L cnet2_m3w2b; do
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=4 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994200 400 m3w_${W}_m2rB > ../data/m3w_${W}_m2rB.out 2>&1 < /dev/null &
  for O in cnet2_m2s2stand cnet2_m2so s1_net; do
    AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=2 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 m3w_${W}_$O > ../data/m3w_${W}_$O.out 2>&1 < /dev/null &
  done
done
wait; echo DONE > ../data/m3w600.done
