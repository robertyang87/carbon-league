#!/bin/bash
# temperature comparison: Grove1 sampled at T (all units, whole game) vs greedy Grove1, seeds 994000 x200; T=0 is the control
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_OPEN_STEPS=300 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for T in 0 0.3 0.5 0.7; do
  AWR_TAU=$T EVO_WORKERS=15 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/g1_greedy.py 994000 200 tmp_T$T > ../data/tmp_T$T.out 2>&1 < /dev/null &
done
wait; echo DONE > ../data/temp.done
