#!/bin/bash
# C1: spot-29 opponent identity + chain status check.
MARK="S29OPPCHECK2312"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/fabler1_cal.status 2>/dev/null || echo NO-STATUS
wc -l /root/carbon/data/bigeval_live/fabler1_m2.jsonl /root/carbon/data/bigeval_live/fabler1_echo2.jsonl 2>/dev/null
echo "bigeval procs: $(pgrep -fc 'bigeval.py exp/' || echo 0)"
echo "--- opp_master2r.py ---"
md5sum /root/carbon/code/exp/opp_master2r.py
cat /root/carbon/code/exp/opp_master2r.py
echo "--- echo2.py head ---"
head -30 /root/carbon/code/exp/echo2.py
echo "--- fabler1.py ---"
cat /root/carbon/code/exp/fabler1.py
echo "${MARK}_END"
