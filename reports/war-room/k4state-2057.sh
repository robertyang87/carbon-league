#!/bin/bash
# h23 direct state probe: did the K4 probe chain launch? Marker: K4STATE2057M
MARK="K4STATE2057M"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- k4probe.status ---"
cat /root/carbon/data/k4probe.status 2>&1 | head -8
echo "--- wrappers ---"
md5sum /root/carbon/code/exp/k4v2r1.py /root/carbon/code/exp/k4v2g1.py 2>&1
echo "--- bigeval procs ---"
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep | head -4
echo "proc count: $(ps -eo args | grep -c '[b]igeval.py exp/')"
echo "--- tag files ---"
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>&1
echo "--- launcher log tails (last 3 lines each) ---"
tail -n 3 /root/carbon/data/k4v2r1_e0.launcher.log 2>&1
tail -n 3 /root/carbon/data/k4v2g1_e0.launcher.log 2>&1
cat /proc/loadavg
echo "${MARK}_END"
