#!/bin/bash
# decision 6: J1c phase-stitched groups b..e (row 2 early, row 3/4/5 late) vs Master2 / opp_m3echo / M-style / opp_w3b, 200 each
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_top_j1cL.npz
declare -A C=( [b]='"TEACHER": 2, "TEACHER_LATE": 3, "LATE_STEP": 200' [c]='"TEACHER": 2, "TEACHER_LATE": 3, "LATE_STEP": 240' [d]='"TEACHER": 2, "TEACHER_LATE": 4, "LATE_STEP": 200' [e]='"TEACHER": 2, "TEACHER_LATE": 5, "LATE_STEP": 200' )
for g in b c d e; do for O in opp_master2r opp_m3echo cnet2_m2so opp_w3b; do
  AWR_US_CONF="{${C[$g]}}" EVO_WORKERS=4 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 j1c${g}_$O > ../data/j1c${g}_$O.out 2>&1 < /dev/null &
done; done
wait; echo DONE > ../data/j1cs.done
