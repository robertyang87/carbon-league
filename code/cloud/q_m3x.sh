#!/bin/bash
# pure-net paired exam of Master3s-planter versions vs Grove1 pure (sp1x_old_*, seeds 994000 x 200, same side) after the B-line test.
# usage: q_m3x.sh TAG WEIGHTS_STEM...
cd /root/carbon/code
until [ -f ../data/bl1.done ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
G=$1; shift
for W in "$@"; do for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 ${G}_${W}_$O > ../data/${G}_${W}_$O.out 2>&1 < /dev/null &
done; done
wait; echo M3X_DONE > ../data/${G}.done
