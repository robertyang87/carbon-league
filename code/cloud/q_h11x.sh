#!/bin/bash
# host 11 (2026-11-11 02:3x): after the round-2 games, paired pure-net exam of the collector-anchored round 1
# (cnet2_b128sp1b) vs the old net, same seeds as the L20 exam (994000 x 200), both arms on this machine.
cd /root/carbon/code
until grep -q ALL_DONE ../data/sp2_scout1.out 2>/dev/null; do sleep 20; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for W in newb:cnet2_b128sp1b old:cnet2_b128m2t3; do w=${W#*:}; v=${W%%:*}
  for O in cnet2_m2s2stand cnet2_m2so s1_net cnet2_m2sstand; do
    AWR_US_WEIGHTS=/root/carbon/data/weights/$w.npz EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 sp1x_${v}_${O}_h11 > /dev/null 2>&1 < /dev/null &
  done
done
wait; echo EXAM_DONE > ../data/sp1x_h11_done
