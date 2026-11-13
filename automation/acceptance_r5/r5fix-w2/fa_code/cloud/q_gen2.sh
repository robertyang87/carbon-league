#!/bin/bash
# plan_path_rl 3.4: Grove1 pure net (greedy) vs the 3 training opps, 5000 seeds each, whole games (imitation data for layer 2)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=300 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for O in m2s:cnet2_m2s2stand m2so:cnet2_m2so s1:s1_net; do n=${O%%:*}; f=${O#*:}
  AWR_TAG=g2$n ../.venv/bin/python awr_gen.py ../data/gen2_$n awr_us_env.py exp/$f.py 6100000 5000 0 0 $(getconf _NPROCESSORS_ONLN) > ../data/gen2_$n.out 2>&1
done
echo DONE > ../data/gen2.done
