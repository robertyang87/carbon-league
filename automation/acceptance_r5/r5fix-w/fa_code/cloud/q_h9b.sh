#!/bin/bash
# host 9, part 2: the other 7500 whole games (training opponents: Master2s stand-in v2, M-style opening net m2so, Scout1 net)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=300 AWR_OPEN_STEPS=300 AWR_KINDS=planter,base
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
until [ -f ../data/sp1_grove1/awr.jsonl ]; do sleep 30; done
TAU=$(awk '{print $3}' ../data/sp_tau_chosen.txt)
for O in m2s:cnet2_m2s2stand m2so:cnet2_m2so scout1:s1_net; do n=${O%%:*}; f=${O#*:}
  ../.venv/bin/python awr_gen.py ../data/sp1_$n awr_us_env.py exp/$f.py 3000000 500 4 $TAU 62 > ../data/sp1_$n.out 2>&1
done
echo ALL_DONE >> ../data/sp1_scout1.out
