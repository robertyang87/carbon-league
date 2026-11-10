#!/bin/bash
# remote.sh TAG N OPP CFGJSON [WORKERS] : run bigeval m2bot vs OPP on host 4 in background
K=$HOME/.ssh/carbon_aliyun_ed25519
ssh -i $K root@47.117.170.99 "cd /root/carbon/code; M2BOT_CFG='$4' OMP_NUM_THREADS=1 BC_W=64 EVO_WORKERS=${5:-20} nohup ../.venv/bin/python bigeval.py exp/m2bot.py $3 998000 $2 $1 > /root/carbon/data/$1.out 2>&1 &"
