#!/bin/bash
# snipe_b ckpt 10 (row 2 all game) vs held-out W3 best (opp_w3b) and Master2 real; E0 vs opp_w3b as the pair baseline. 200 each, seeds 994000
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_snipeb_10.npz EVO_WORKERS=16 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_w3b.py 994000 200 snp10_w3b > ../data/snp10_w3b.out 2>&1 < /dev/null &
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_snipeb_10.npz EVO_WORKERS=16 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_master2r.py 994000 200 snp10_m2r > ../data/snp10_m2r.out 2>&1 < /dev/null &
AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz EVO_WORKERS=16 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/opp_w3b.py 994000 200 e0_w3b > ../data/e0_w3b.out 2>&1 < /dev/null &
wait; echo DONE > ../data/snp10.done
