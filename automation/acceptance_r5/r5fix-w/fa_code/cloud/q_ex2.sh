#!/bin/bash
# search-labelled games for distillation (72f5ef4/cd2a72e): our side = the measured multi-car pack fx_m2t3S20q3 (md5 ad69165e),
# greedy only (K=0), whole games. usage: q_ex1.sh SEED0 NSEEDS  -> ../data/ex1_<opp> per opponent, tag ex1<opp>
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=240
for O in m2s:cnet2_m2s2stand m2so:cnet2_m2so scout1:s1_net; do n=${O%%:*}; f=${O#*:}
  AWR_TAG=ex2$n ../.venv/bin/python awr_gen.py ../data/ex2_$n exp/fx_m2t3S20q3.py exp/$f.py $1 $2 0 0 $(getconf _NPROCESSORS_ONLN) > ../data/ex2_$n.out 2>&1
done
echo ALL_DONE > ../data/ex2_$1.done
