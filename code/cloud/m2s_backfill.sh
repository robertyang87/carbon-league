#!/bin/bash
cd ~/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=16 CNET2_TEACHER=7
../.venv/bin/python bigeval.py exp/cnet2_v5t7g.py exp/cnet2_m2stand.py 994000 50 c_v5t7g_m2s > /dev/null 2>&1
for f in 5 10 15; do RL_W=$(pwd)/../data/rlc2/eval/w_$f.npz ../.venv/bin/python bigeval.py exp/cnet2_rlg.py exp/cnet2_m2stand.py 994000 50 rl_${f}_m2s > /dev/null 2>&1; done
RL_W=$(pwd)/../data/rlc2/eval/w_15.npz ../.venv/bin/python bk_check.py exp/cnet2_rlg.py exp/submission_cnet2_v13.py rl_15_v13 > ../data/rl15_bk.txt 2>&1
echo BACKFILL_DONE >> ../data/rl15_bk.txt
