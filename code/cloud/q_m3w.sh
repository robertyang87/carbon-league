#!/bin/bash
# runbook 4: screen 6 Master3s replicas, pure net vs Master2 real, 200 each (seeds 994000), pairs with m2r_pure / m2r_pure600
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
L="cnet2_m3w1b cnet2_m3w1L cnet2_m3w2b cnet2_m3w2L cnet2_m3w3b cnet2_m3w3L"
for W in $L; do until [ -f ../data/weights/$W.npz.ok ]; do sleep 20; done; done
for W in $L; do
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 200 m3w_${W}_m2r > ../data/m3w_${W}_m2r.out 2>&1 < /dev/null &
done
wait; echo DONE > ../data/m3w.done
