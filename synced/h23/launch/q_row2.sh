#!/bin/bash
# Grove1 pure with style row 2 for the whole game (TEACHER_LATE=2) vs the original (7 after step 240): pairs with m2r_pure / sp1x_old_*
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}' AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz
EVO_WORKERS=16 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 200 row2_m2r > ../data/row2_m2r.out 2>&1 < /dev/null &
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 row2_$O > ../data/row2_$O.out 2>&1 < /dev/null &
done
wait; echo DONE > ../data/row2.done
