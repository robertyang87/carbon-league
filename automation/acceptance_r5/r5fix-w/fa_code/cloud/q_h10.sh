#!/bin/bash
# host 10 (2026-11-10 night): resume sp1_scout1 (directory copied from the Mac; awr_gen skips the games already there)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=300 AWR_OPEN_STEPS=300 AWR_KINDS=planter,base
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
until [ -f ../data/sp1_scout1/READY ]; do sleep 15; done
../.venv/bin/python awr_gen.py ../data/sp1_scout1 awr_us_env.py exp/s1_net.py 3000000 500 4 0.3 62 > ../data/sp1_scout1.out 2>&1
