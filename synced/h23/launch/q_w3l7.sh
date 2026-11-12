#!/bin/bash
# candidate W3L with the Grove1 late switch (row 7 after step 240) vs Master2 real, 200 (pairs with m3w_cnet2_m3w3L_m2r = row 2 all game)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz
../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 200 m3w_w3L_late7_m2r > ../data/w3l7.out 2>&1
echo DONE > ../data/w3l7.done
