#!/bin/bash
# W6L exam: pure net, row 2 all game, vs Master2 real / opp_m3echo / M-style, 200 each (pairs with plm_E0_*)
cd /root/carbon/code
until [ -f ../data/weights/cnet2_m3w6L.npz.ok ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}' AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w6L.npz
for O in opp_master2r opp_m3echo cnet2_m2so; do EVO_WORKERS=14 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 w6_$O > ../data/w6_$O.out 2>&1 < /dev/null & done
wait; echo DONE > ../data/w6.done
