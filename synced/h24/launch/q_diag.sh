#!/bin/bash
# (a) sampled-vs-greedy diagnostic: ckpt15 and E0, our side (0.3 cars, 0.5 planters/base) vs W3best sampled 0.5, 200 each
# (b) int8 vs f16 floor: W5 f16 no-roll pack vs Master2 x800; W5 int8 pack +600 (seeds 994200) to reach 800
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
C='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240, "OPEN_T": [0.3, 0.5, 0.5]}'
AWR_OPEN_STEPS=300 AWR_US_CONF="$C" AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_snipeb_15.npz EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_w3b_t05.py 994000 200 diag_s15 > ../data/diag_s15.out 2>&1 < /dev/null &
AWR_OPEN_STEPS=300 AWR_US_CONF="$C" AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_w3b_t05.py 994000 200 diag_e0 > ../data/diag_e0.out 2>&1 < /dev/null &
EVO_WORKERS=24 setsid nohup ../.venv/bin/python bigeval.py exp/submission_echo2_w5f16.py exp/opp_master2r.py 994000 800 q16_w5_m2r > ../data/q16.out 2>&1 < /dev/null &
EVO_WORKERS=16 setsid nohup ../.venv/bin/python bigeval.py exp/submission_echo2_w5q8.py exp/opp_master2r.py 994200 600 w5pkB_m2r > ../data/w5pkB.out 2>&1 < /dev/null &
wait; echo DONE > ../data/diag.done
