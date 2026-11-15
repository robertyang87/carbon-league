#!/bin/bash
# spot25 step F3: one-shot smoke status read.
MARK="SPOT25F8K2"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- PROC ---"
ps -eo pid,etime,args | grep bigeval | grep -v grep || echo "NO-BIGEVAL-PROC"
echo "--- LIVE LINES ---"
wc -l /root/carbon/data/bigeval_live/spot25_smoke.jsonl 2>/dev/null || echo "NO-LIVE-FILE"
cat /root/carbon/data/bigeval_live/spot25_smoke.jsonl 2>/dev/null
echo "--- SMOKE OUT TAIL ---"
tail -8 /root/carbon/data/spot25_smoke.out 2>/dev/null
echo "--- SUMMARY LINE ---"
grep 'spot25_smoke' /root/carbon/data/bigeval.jsonl 2>/dev/null || echo "NO-SUMMARY-YET"
echo "MARKER $MARK END"
