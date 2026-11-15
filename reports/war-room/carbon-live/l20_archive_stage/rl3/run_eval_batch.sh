#!/bin/bash
cd /root/carbon/code
export CNET2_TEACHER=2 RL2_SNAP_TEMP=0.5 CNET2_TEMP_KIND=0.15,0.3,0.3 RL2_SERVERS=1 RL2_SERVER_MAXB=1024 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/carbon/.venv/bin/python
$PY rl2_ppo_eval.py --adapter cnet2 --w /root/carbon/data/rl3/snipe_n1/ckpt_75.npz --opps exp/opp_master2r.py,exp/opp_m3echo.py --pairs 32 --workers 8 --tag ctrl_it75_env --log /root/carbon/data/rl3/snipe_n1/offline_eval_batch.jsonl
$PY rl2_ppo_eval.py --adapter cnet2 --w /root/carbon/data/rl3/snipe_n1/ckpt_85_frozen.npz --opps exp/opp_master2r.py,exp/opp_m3echo.py --pairs 32 --workers 8 --tag it85_env --log /root/carbon/data/rl3/snipe_n1/offline_eval_batch.jsonl
echo BATCH_DONE
