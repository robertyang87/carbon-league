#!/bin/bash
# J1 phase-stitched exams (runbook_echo2 3f): groups given as args (a..e); vs Master2 / opp_m3echo / M-style / opp_w3b, 200 each
cd /root/carbon/code
until [ -f ../data/weights/cnet2_top_j1L.npz.ok ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_top_j1L.npz
declare -A C=( [a]='"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240' [b]='"TEACHER": 2, "TEACHER_LATE": 3, "LATE_STEP": 200' [c]='"TEACHER": 2, "TEACHER_LATE": 3, "LATE_STEP": 240' [d]='"TEACHER": 2, "TEACHER_LATE": 4, "LATE_STEP": 200' [e]='"TEACHER": 2, "TEACHER_LATE": 5, "LATE_STEP": 200' )
for g in "$@"; do for O in opp_master2r opp_m3echo cnet2_m2so opp_w3b; do
  AWR_US_CONF="{${C[$g]}}" EVO_WORKERS=6 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 j1${g}_$O > ../data/j1${g}_$O.out 2>&1 < /dev/null &
done; done
wait; echo DONE > ../data/j1s_$1.done
