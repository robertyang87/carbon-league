#!/bin/bash
# kill the 1-worker generation (nproc honoured OMP_NUM_THREADS=1) and restart with all cores; awr_gen resumes from awr_raw.jsonl
for p in $(ps -eo pid,args | awk '/bash \/root\/q_ex1/ && !/awk/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,args | awk '/awr_gen.py ..\/data\/ex1_/ && !/awk/ {print $1}'); do kill $p; done
sleep 2
setsid nohup bash /root/q_ex1.sh $1 167 > /root/q_ex1.log 2>&1 < /dev/null &
