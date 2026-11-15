#!/bin/bash
# h23 status (read-only): Echo2 anchor chain final state + summaries.
MARK="H23ANCHORSTATUS2011D"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/echo2_anchor.status 2>&1
echo "--- NO BIGEVAL RUNNING? ---"
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep || echo "NO_BIGEVAL_RUNNING"
echo "--- LIVE LINE COUNTS ---"
wc -l /root/carbon/data/bigeval_live/echo2_m2.jsonl /root/carbon/data/bigeval_live/m3s2g1_echo2.jsonl 2>&1
echo "--- SUMMARIES (data/bigeval.jsonl) ---"
grep -E '"tag": "(echo2_m2|m3s2g1_echo2)"' /root/carbon/data/bigeval.jsonl 2>&1 || grep -E 'echo2_m2|m3s2g1_echo2' /root/carbon/data/bigeval.jsonl 2>&1
echo "--- LAUNCHER LOG TAILS ---"
tail -3 /root/carbon/data/echo2_m2.launcher.log 2>&1
tail -3 /root/carbon/data/m3s2g1_echo2.launcher.log 2>&1
echo "${MARK}_END"
