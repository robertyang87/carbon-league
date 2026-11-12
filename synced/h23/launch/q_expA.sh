#!/bin/bash
# experiment A: P2/P3/P4 vs E0 head-to-head; experiment B (vs Master2): react-only 0.5 / 1.0. seeds 994000 x200
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for P in p2 p3 p4; do EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py exp/echo_$P.py exp/echo_e0.py 994000 200 expA_$P > ../data/expA_$P.out 2>&1 < /dev/null & done
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz
for R in 0.5 1.0; do AWR_US_CONF="{\"TEACHER\": 2, \"TEACHER_LATE\": 2, \"LATE_STEP\": 240, \"MACRO\": {\"PL_MIN_E\": 0, \"PL_MIN_M\": 0, \"PL_MIN_REACT\": $R}}" EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 200 expB_R${R}_m2r > ../data/expB_$R.out 2>&1 < /dev/null & done
wait; echo DONE > ../data/expAB.done
