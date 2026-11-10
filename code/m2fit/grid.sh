#!/bin/bash
# grid.sh N OPP TAG1 CFG1 TAG2 CFG2 ... : sequential bigeval runs on host 4 (20 workers), detached; CFG = plain json
K=$HOME/.ssh/carbon_aliyun_ed25519
N=$1; OPP=$2; shift 2
CMD="cd /root/carbon/code"
while [ $# -gt 1 ]; do
  B=$(printf '%s' "$2" | base64 | tr -d '\n')
  CMD="$CMD; M2BOT_CFG_B64=$B OMP_NUM_THREADS=1 BC_W=64 EVO_WORKERS=${W:-20} ../.venv/bin/python bigeval.py exp/m2bot.py $OPP 998000 $N $1 > /root/carbon/data/$1.out 2>&1"
  shift 2
done
ssh -i $K root@47.117.170.99 "nohup bash -c '$CMD' > /dev/null 2>&1 &"
