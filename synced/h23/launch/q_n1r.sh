#!/bin/bash
# advice 01:50 #2: n1r int8 pack: 600 vs Master2 / opp_w3b; 200 vs M-style / m3echo / E0 / snipe_b_30 / opp_m6r; E0 baseline vs opp_w3b seeds 994200x200
cd /root/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; P=../data/packs/n1r_int8.py
r() { setsid nohup ../.venv/bin/python bigeval.py $1 exp/$2.py $3 $4 $5 > ../data/$5.out 2>&1 < /dev/null & }
EVO_WORKERS=10 r $P opp_master2r 994000 600 pkn1r_opp_master2r
EVO_WORKERS=10 r $P opp_w3b 994000 600 pkn1r_opp_w3b
for O in cnet2_m2so opp_m3echo echo_e0 sb30 opp_m6r; do EVO_WORKERS=7 r $P $O 994000 200 pkn1r_$O; done
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}' EVO_WORKERS=6 r awr_us_env.py opp_w3b 994200 200 e0_w3bD
wait; echo DONE > ../data/n1r.done
