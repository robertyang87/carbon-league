#!/bin/bash
# L20 CPUs (2026-11-11 02:3x): acceptance exam of the first AWR round, pure net (no search), greedy.
# new = cnet2_b128sp1 vs old = cnet2_b128m2t3, same seeds/sides -> paired; 4 opponents (3 training + exam-only m2sstand v1).
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for W in new:cnet2_b128sp1 old:cnet2_b128m2t3; do w=${W#*:}; v=${W%%:*}
  for O in cnet2_m2s2stand cnet2_m2so s1_net cnet2_m2sstand; do
    AWR_US_WEIGHTS=/root/carbon/data/weights/$w.npz EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 sp1x_${v}_$O > ../data/sp1x_${v}_$O.out 2>&1 < /dev/null &
  done
done
wait; echo EXAM_DONE > ../data/sp1x_done
