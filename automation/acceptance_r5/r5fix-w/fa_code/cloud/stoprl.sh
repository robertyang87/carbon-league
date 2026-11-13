#!/bin/bash
pkill -f "rl2_actor.py"; pkill -f "cloud/rl_evald.sh"; sleep 3
ps -eo pid,ppid,args | grep "[m]ultiprocessing.spawn" | awk '$2==1 {print $1}' | xargs -r kill
sleep 1; uptime
