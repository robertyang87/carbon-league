#!/bin/bash
# W5L to 600 pairs: +400 vs Master2 (pairs with m3w_cnet2_m3w3L_m2rB) and +400 vs opp_m3echo (with E0 on the same seeds)
cd /root/carbon/code
until [ -f ../data/ens.done ] && [ -f ../data/e2w.done ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w5L.npz EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994200 400 w5x_m2rB > ../data/w5x1.out 2>&1 < /dev/null &
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w5L.npz EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_m3echo.py 994200 400 w5x_echoB > ../data/w5x2.out 2>&1 < /dev/null &
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_m3echo.py 994200 400 e0x_echoB > ../data/w5x3.out 2>&1 < /dev/null &
wait; echo DONE > ../data/w5x.done
