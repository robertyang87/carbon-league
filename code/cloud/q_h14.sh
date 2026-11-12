#!/bin/bash
# host 14: finish the search-vs-pure exam (first 100 seeds per group, resumed), then car>2nd forks (d9d0d0a)
cd /root/carbon/code
until [ -f exp/fx_m2t3S20.py ]; do sleep 10; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
miss() { python3 -c "
import json,os,sys
f='../data/bigeval_live/$1.jsonl'; d={json.loads(l)['seed'] for l in open(f)} if os.path.exists(f) else set()
print(','.join(str(s) for s in range($2,$2+100) if s not in d))"; }
for O in cnet2_m2s2stand cnet2_m2so; do
  EVO_SEEDS=$(miss sx_S20_$O 994000) EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20.py exp/$O.py 0 0 sx_S20_$O > ../data/sx_S20_$O.out 2>&1 < /dev/null &
done
EVO_SEEDS=$(miss sx_S20_vs_pure 995000) AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' EVO_WORKERS=21 \
  setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20.py awr_us_env.py 0 0 sx_S20_vs_pure > ../data/sx_S20_vs_pure.out 2>&1 < /dev/null &
wait; echo SX_DONE > ../data/sx.done
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 62 > ../data/fkcar_$O.out 2>&1
done
echo ALL_DONE > ../data/fkcar.done
