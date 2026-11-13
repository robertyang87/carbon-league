#!/bin/bash
# after m3z: stage-A pure-net exam (aux1_last) + Master2 real baselines (Grove1 pure, Grove1 search) 200 each
cd /root/carbon/code
until [ -f ../data/weights/cnet2_b128aux1L.npz.ok ] && [ -f exp/master2_src.py ]; do sleep 20; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
W=cnet2_b128aux1L
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 auxA_${W}_$O > ../data/auxA_$O.out 2>&1 < /dev/null &
done
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 200 m2r_pure > ../data/m2r_pure.out 2>&1 < /dev/null &
EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20m.py exp/opp_master2r.py 994000 200 m2r_S20m > ../data/m2r_S20m.out 2>&1 < /dev/null &
wait; echo DONE > ../data/h22.done
