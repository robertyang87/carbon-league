#!/bin/bash
# L20-3 CPUs (2026-11-11 03:3x): pure-net exam of the soft-anchored models, same seeds/sides as the first L20 exam
# (994000 x 200, 4 opponents); the old-net arm is reused from that exam (pure-net games are bit-identical across machines, 698/698).
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for k in 30 100; do
  until [ -f ../data/K${k}_DONE ] || grep -q "== build rounds" ../data/train_k30.log; do sleep 15; done
  [ -f ../data/K${k}_DONE ] || break
  for O in cnet2_m2s2stand cnet2_m2so s1_net cnet2_m2sstand; do
    AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128sp1k$k.npz EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 sp1x_k${k}_$O > /dev/null 2>&1 < /dev/null &
  done
  wait; touch ../data/EXAM_K${k}_DONE
done
