#!/bin/bash
# runbook_echo2 3g: L variants on Echo1 (W3L, row 2 all game) vs Master2 real / opp_m3echo / M-style / opp_w3b, 200 each (pairs with plm_E0_*, e0_w3b)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz
B='"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240'
declare -A C=( [L1]=', "LATE_PLANT": {}' [L2]=', "LATE_PLANT": {"FROM": 200}, "MACRO": {"PL_MIN_L": 2, "PL_T2": 200, "PUSH_STOP": 285}' [L3]=', "LATE_PLANT": {"FROM": 200}, "MACRO": {"PL_MIN_L": 3, "PL_T2": 200, "PUSH_STOP": 285}' )
for v in "$@"; do for O in opp_master2r opp_m3echo cnet2_m2so opp_w3b; do
  AWR_US_CONF="{$B${C[$v]}}" EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 lp_${v}_$O > ../data/lp_${v}_$O.out 2>&1 < /dev/null &
done; done
wait; echo DONE > ../data/lp_$1.done
