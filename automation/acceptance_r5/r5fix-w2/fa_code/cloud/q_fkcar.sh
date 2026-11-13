#!/bin/bash
# host 13: car>2nd forks (code d9d0d0a) after the search-vs-pure exam; Grove1 pure net vs the 3 training opponents
cd /root/carbon/code
while pgrep -f "[b]igeval.py exp/fx_m2t3S20" > /dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 62 > ../data/fkcar_$O.out 2>&1
done
echo ALL_DONE > ../data/fkcar.done
