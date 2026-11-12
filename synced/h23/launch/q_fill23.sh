#!/bin/bash
# filler on host 23 after the H exams: W6 to 600 pairs (+400 vs Master2 and opp_m3echo, seeds 994200; baselines m3w_cnet2_m3w3L_m2rB / e0x_echoB)
cd /root/carbon/code
until [ -f ../data/hs.done ] && [ -f ../data/snp10.done ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}' AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w6L.npz
EVO_WORKERS=30 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994200 400 w6x_m2rB > ../data/w6x1.out 2>&1 < /dev/null &
EVO_WORKERS=30 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_m3echo.py 994200 400 w6x_echoB > ../data/w6x2.out 2>&1 < /dev/null &
wait; echo DONE > ../data/w6x.done
