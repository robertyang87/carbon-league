#!/bin/bash
# host 5 (2026-11-10): opening-information diagnostics vs the Master2s stand-in, own seeds, paired within this machine
cd /root/carbon/code; export OMP_NUM_THREADS=1
W=12   # fixed: nproc obeys OMP_NUM_THREADS=1
EVO_WORKERS=$W setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20d.py exp/cnet2_m2s2stand.py 993500 150 m2s_G20_h5 > ../data/m2s_G20_h5.out 2>&1 < /dev/null &
for v in OR20:EVAL_ORACLE_STEPS=20 OR30:EVAL_ORACLE_STEPS=30 OR40:EVAL_ORACLE_STEPS=40 SNK:EVAL_SNAKE=8:15; do n=${v%%:*}; e=${v#*:}
  env $e EVO_WORKERS=$W setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20d.py exp/cnet2_m2s2stand.py 993500 150 m2s_${n}_h5 > ../data/m2s_${n}_h5.out 2>&1 < /dev/null &
done
wait
