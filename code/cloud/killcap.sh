#!/bin/bash
pkill -f "cloud/cap.sh"; pkill -f "bigeval.py exp/cnet2_v7s_mfxc"; sleep 3
ps -eo pid,ppid,args | grep "[m]ultiprocessing\|[c]oncurrent" | awk '$2==1 {print $1}' | xargs -r kill 2>/dev/null; sleep 1; uptime
