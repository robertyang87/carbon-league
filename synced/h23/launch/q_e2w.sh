#!/bin/bash
# runbook_echo2 sec.1 exam: W5L, W4L pure net (row 2 all game) vs Master2 real, opp_m3echo, M-style; 200 each, seeds 994000 (pairs with plm_E0_*)
cd /root/carbon/code
for W in cnet2_m3w5L cnet2_m3w4L; do until [ -f ../data/weights/$W.npz.ok ]; do sleep 30; done; done
until [ -f ../data/echo3.done ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
for W in cnet2_m3w5L cnet2_m3w4L; do for O in opp_master2r opp_m3echo cnet2_m2so; do
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 e2w_${W}_$O > ../data/e2w_${W}_$O.out 2>&1 < /dev/null &
done; done
wait; echo DONE > ../data/e2w.done
