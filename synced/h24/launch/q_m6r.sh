#!/bin/bash
# advice 01:50 #4: opp_m6r as fixed examiner: E0 and snipe_b_30 vs opp_m6r 200 each (pure net row 2); n1r pack timing; J1c-d to 600 (vs Master2/m3echo/w3b, seeds 994200x400)
cd /root/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
r() { setsid nohup ../.venv/bin/python bigeval.py $1 exp/$2.py $3 $4 $5 > ../data/$5.out 2>&1 < /dev/null & }
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz EVO_WORKERS=10 r awr_us_env.py opp_m6r 994000 200 e0_opp_m6r
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_snipe_b_30.npz EVO_WORKERS=10 r awr_us_env.py opp_m6r 994000 200 sb30_opp_m6r
for O in opp_master2r opp_m3echo opp_w3b; do AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_top_j1cL.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 4, "LATE_STEP": 200}' EVO_WORKERS=12 r awr_us_env.py $O 994200 400 j1cdX_$O; done
../.venv/bin/python cloud/pack_timing.py ../data/packs/n1r_int8.py 10 > ../data/pkn1r_timing.out 2>&1
wait; echo DONE > ../data/m6r.done
