#!/bin/bash
pkill -f "/root/nocap.sh"; pkill -f "bigeval.py exp/submission_cnet2_v16s_nocap.py exp/opp_1cz_v9"; sleep 3
ps -eo pid,ppid,args | grep "[m]ultiprocessing\|[c]oncurrent" | awk '$2==1 {print $1}' | xargs -r kill 2>/dev/null
echo DONE > /root/carbon/data/nocap.done; sleep 5; uptime
