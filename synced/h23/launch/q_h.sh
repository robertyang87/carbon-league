#!/bin/bash
# H1-H3 exam: pure net, row 2 all game, vs Master2 real / opp_m3echo / M-style, 200 each (pairs with plm_E0_*)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
for h in h1 h2 h3; do ( until [ -f ../data/weights/cnet2_m3${h}L.npz.ok ]; do sleep 30; done
  for O in opp_master2r opp_m3echo cnet2_m2so; do AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3${h}L.npz EVO_WORKERS=7 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 hs_${h}_$O > ../data/hs_${h}_$O.out 2>&1 < /dev/null & done; wait ) & done
wait; echo DONE > ../data/hs.done
