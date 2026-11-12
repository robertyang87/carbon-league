#!/bin/bash
# host 12 (2026-11-11 03:xx): CPU makes games only. Round-3 whole games vs the three training opponents, seeds 3200000+,
# tagged per opponent. Same Grove1 policy as rounds 1-2, so the corpora pool for any AWR recipe.
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=300 AWR_OPEN_STEPS=300 AWR_KINDS=planter,base
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for O in m2s:cnet2_m2s2stand m2so:cnet2_m2so scout1:s1_net; do n=${O%%:*}; f=${O#*:}
  AWR_TAG=sp3$n ../.venv/bin/python awr_gen.py ../data/sp3_$n awr_us_env.py exp/$f.py 3200000 500 4 0.3 62 > ../data/sp3_$n.out 2>&1
done
echo ALL_DONE >> ../data/sp3_scout1.out
