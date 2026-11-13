#!/bin/bash
pkill -f "cloud/rs.sh"; pkill -f "cloud/rs8.sh"; pkill -f "cloud/rs10.sh"; pkill -f "bigeval.py exp/cnet2_v7s"; sleep 3
ps -eo pid,ppid,args | grep "[m]ultiprocessing\|[c]oncurrent" | awk '$2==1 {print $1}' | xargs -r kill 2>/dev/null; sleep 1
cd ~/carbon/code; cp .cnet2_search_next.py cnet2_search.py; cp .eval_v53_next.py eval_v53.py; cp .bigeval_next.py bigeval.py
rm -f ~/carbon/data/bigeval_live/c_v7s_r6_v13.jsonl ~/carbon/data/bigeval_live/c_v7s8_*; uptime
