#!/bin/bash
# decision 2: snipe_b_30 passed (w3b +176±63, Master2 +124±43) -> other examiners, 200 each, pure net row 2, same seeds as E0 baselines; plus head-to-head vs echo_e0
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_snipe_b_30.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
for O in opp_m3echo cnet2_m2so opp_1cz_v9 echo_e0 g1_pure; do
  EVO_WORKERS=6 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 ck30x_$O > ../data/ck30x_$O.out 2>&1 < /dev/null &
done
wait; echo DONE > ../data/ck30x.done
