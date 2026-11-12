#!/bin/bash
# host 11 (2026-11-11 01:5x): CPU makes games only (training moved to the L20). Round-2 whole games vs the three
# training opponents, new seeds 3100000+, tagged per opponent so corpora never collide. No Grove1 mirror opponent.
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=300 AWR_OPEN_STEPS=300 AWR_KINDS=planter,base
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for O in m2s:cnet2_m2s2stand m2so:cnet2_m2so scout1:s1_net; do n=${O%%:*}; f=${O#*:}
  AWR_TAG=sp2$n ../.venv/bin/python awr_gen.py ../data/sp2_$n awr_us_env.py exp/$f.py 3100000 500 4 0.3 62 > ../data/sp2_$n.out 2>&1
done
echo ALL_DONE >> ../data/sp2_scout1.out
