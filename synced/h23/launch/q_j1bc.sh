#!/bin/bash
# J1b / J1c, group a (Master3s row all game): vs Master2 real / opp_m3echo / opp_w3b, 200 each
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
for v in j1b j1c; do ( until [ -f ../data/weights/cnet2_top_${v}L.npz.ok ]; do sleep 30; done
  for O in opp_master2r opp_m3echo opp_w3b; do AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_top_${v}L.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 ${v}a_$O > ../data/${v}a_$O.out 2>&1 < /dev/null & done; wait ) & done
wait; echo DONE > ../data/j1bc.done
