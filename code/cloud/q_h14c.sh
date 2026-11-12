#!/bin/bash
# host 14 (restart so the records carry seq_changed): multi-car search SEQ_KEYS=3 -> car>2nd forks -> 1-car 50-sim control
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rm -f ../data/bigeval_live/sx_Q3_*.jsonl
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20q3.py exp/$O.py 994000 100 sx_Q3_$O > ../data/sx_Q3_$O.out 2>&1 < /dev/null &
done
wait; echo Q3_DONE > ../data/q3.done
( export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
  export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
  for O in cnet2_m2s2stand cnet2_m2so s1_net; do
    FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 62 > ../data/fkcar_$O.out 2>&1
  done
  echo ALL_DONE > ../data/fkcar.done )
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S50n.py exp/$O.py 994000 50 sx_S50_$O > ../data/sx_S50_$O.out 2>&1 < /dev/null &
done
wait; echo S50_DONE > ../data/s50.done
