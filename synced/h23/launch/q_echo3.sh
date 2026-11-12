#!/bin/bash
# after the 600-pair exams: three 200-game matches against the frozen opp_m3echo (seeds 994000)
cd /root/carbon/code
until [ -f ../data/m3w600.done ] && [ -f ../data/w3l7.done ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py exp/g1_pure.py exp/opp_m3echo.py 994000 200 echo_g1pure > ../data/echo_g1pure.out 2>&1 < /dev/null &
EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py exp/submission_cnet2_m3w3L_l2.py exp/opp_m3echo.py 994000 200 echo_w3pack > ../data/echo_w3pack.out 2>&1 < /dev/null &
EVO_WORKERS=20 setsid nohup ../.venv/bin/python bigeval.py exp/opp_master2r.py exp/opp_m3echo.py 994000 200 echo_m2r > ../data/echo_m2r.out 2>&1 < /dev/null &
wait; echo DONE > ../data/echo3.done
