#!/bin/bash
MARK=S30E3SUM0303
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
echo "--- LAUNCH-LOG-FULL ---"
cat /root/carbon/data/echo3_m2.launch.log
echo "--- LIVE ---"
wc -l /root/carbon/data/bigeval_live/echo3_m2.jsonl
md5sum /root/carbon/data/bigeval_live/echo3_m2.jsonl
ls -l --time-style=full-iso /root/carbon/data/bigeval_live/echo3_m2.jsonl
echo -n "bigeval-procs: "; pgrep -cf "bigeval.py" || echo 0
echo "$MARK PAYLOAD-END"
