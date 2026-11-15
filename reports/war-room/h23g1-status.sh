#!/bin/bash
# h23: post-run status for the g1 chained exams (read-only).
MARK="H23G1STAT1956C"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/g1_chain.status 2>&1
echo "--- CHAIN OUT TAIL ---"
tail -5 /root/carbon/data/g1_chain.out 2>&1
echo "--- ANY BIGEVAL RUNNING? ---"
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep || echo "NO_BIGEVAL_RUNNING"
echo "--- LINE COUNTS ---"
wc -l /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl /root/carbon/data/bigeval_live/m3s2g1_m2r.jsonl /root/carbon/data/bigeval_live/m3s2g1_e0.jsonl 2>&1
echo "--- SUMMARIES ---"
grep 'm3s2g1_m2r' /root/carbon/data/bigeval.jsonl 2>&1 | head -c 700; echo
grep 'm3s2g1_e0' /root/carbon/data/bigeval.jsonl 2>&1 | head -c 700; echo
echo "--- LAUNCHER LOGS ---"
ls -la /root/carbon/data/m3s2g1_m2r.launcher.log /root/carbon/data/m3s2g1_e0.launcher.log 2>&1
tail -3 /root/carbon/data/m3s2g1_m2r.launcher.log 2>&1
tail -3 /root/carbon/data/m3s2g1_e0.launcher.log 2>&1
cat /proc/loadavg
echo "${MARK}_END"
