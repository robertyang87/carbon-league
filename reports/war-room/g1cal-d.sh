#!/bin/bash
# g1cal step D (read-only): confirm first per-game lines landed for m3s2g1_m2.
MARK="G1CALD2M8"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- PROCS ---"
ps -eo pid,ppid,etime,args | grep 'm3s2g1_m2' | grep -v grep || echo "NO-M3S2G1-PROC"
echo "--- LIVE FILE ---"
ls -la /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1
wc -l /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1
head -2 /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1 | cut -c1-800
echo "--- SUMMARY LINE ---"
grep 'm3s2g1_m2' /root/carbon/data/bigeval.jsonl 2>&1 | head -3 || echo NO-SUMMARY-YET
echo "--- OUT TAIL ---"
tail -5 /root/carbon/data/m3s2g1_m2.out 2>&1
echo "MARKER $MARK END"
