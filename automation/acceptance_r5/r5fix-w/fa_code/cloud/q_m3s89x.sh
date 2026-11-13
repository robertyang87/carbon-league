#!/bin/bash
# pure-net exam of the Master3s-89 replicas: 3 training opps x200 + Master2 real x600, seeds 994000
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for W in cnet2_b128m3s89A cnet2_b128m3s89L50; do until [ -f ../data/weights/$W.npz.ok ]; do sleep 20; done; done
for W in cnet2_b128m3s89A cnet2_b128m3s89L50; do
  for O in cnet2_m2s2stand cnet2_m2so s1_net; do
    AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=6 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 m89_${W}_$O > ../data/m89_${W}_$O.out 2>&1 < /dev/null &
  done
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=14 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 600 m89_${W}_m2r > ../data/m89_${W}_m2r.out 2>&1 < /dev/null &
done
wait; echo DONE > ../data/m89.done
