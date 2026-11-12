#!/bin/bash
# host 14 after the resumed search-vs-pure exam: SEQ_KEYS=3 multi-car search (7c4553b source in the pack), same seeds as
# the B group, 3 training opponents x 100; then car>2nd forks
cd /root/carbon/code
while pgrep -f "[b]igeval.py exp/fx_m2t3S20.py" > /dev/null; do sleep 30; done
echo SX_DONE > ../data/sx.done
until [ -f exp/fx_m2t3S20q3.py ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20q3.py exp/$O.py 994000 100 sx_Q3_$O > ../data/sx_Q3_$O.out 2>&1 < /dev/null &
done
wait; echo Q3_DONE > ../data/q3.done
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 62 > ../data/fkcar_$O.out 2>&1
done
echo ALL_DONE > ../data/fkcar.done
