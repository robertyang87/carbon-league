#!/bin/bash
# fork data, base decisions only (2026-11-11 04:2x): does recruiting a collector (0>C, P>C) really pay at 50 steps?
# Grove1 pure net vs the three training opponents, 200 games each, FORK_P 0.3, horizons 10/25/50, 20% played to the end.
# usage: q_fkb.sh SEED0 TAG   (each host its own seed range)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.2 FORK_KINDS="0>C,0>P,C>0,C>P,P>0,P>C"
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  FORK_TAG=$2_$O ../.venv/bin/python fork_gen.py ../data/fkb_$2 awr_us_env.py exp/$O.py $1 200 62 > ../data/fkb_$2_$O.out 2>&1
done
echo ALL_DONE > ../data/fkb_$2.done
