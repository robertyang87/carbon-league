#!/bin/bash
# N-B (decision point 4): N1 replica (attention) vs N0 replica, same seeds; vs Master2 real / opp_m3echo / opp_w3b, 200 each (N0rep vs Master2, m3echo exist)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3n0L.npz EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_w3b.py 994000 200 n0r_opp_w3b > ../data/n0r_w3b.out 2>&1 < /dev/null &
until [ -f ../data/weights/cnet2_m3n1L.npz.ok ]; do sleep 30; done
for O in opp_master2r opp_m3echo opp_w3b; do AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3n1L.npz EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 n1r_$O > ../data/n1r_$O.out 2>&1 < /dev/null & done
wait; echo DONE > ../data/nb.done
