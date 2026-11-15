#!/bin/bash
# spot25 step F2: wait for smoke summary, print verbatim results.
MARK="SPOT25F7K9"
echo "MARKER $MARK START"
date '+%F %T %Z'
for i in 1 2 3 4 5 6 7 8 9 10; do
  if grep -q 'spot25_smoke' /root/carbon/data/bigeval.jsonl 2>/dev/null; then echo "SUMMARY-FOUND at iter $i"; break; fi
  sleep 45
done
echo "--- SMOKE OUT ---"
cat /root/carbon/data/spot25_smoke.out 2>/dev/null
echo "--- LIVE JSONL ---"
cat /root/carbon/data/bigeval_live/spot25_smoke.jsonl 2>/dev/null || echo "NO-LIVE-FILE"
echo "--- SUMMARY LINE ---"
grep 'spot25_smoke' /root/carbon/data/bigeval.jsonl 2>/dev/null || echo "NO-SUMMARY-YET"
echo "--- PROC ---"
ps -eo pid,etime,args | grep bigeval | grep -v grep || echo "NO-BIGEVAL-PROC"
echo "MARKER $MARK END"
