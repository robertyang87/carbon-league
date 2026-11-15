#!/bin/bash
# Runs on h24 via cloud assistant. Launches the 4-game smoke (detached, low workers). Does NOT touch the running E0 field.
MARK="ECHO2PREPSMKH24"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
cd /root/carbon/code
timeout 60 md5sum exp/m3s2r1.py exp/echo2.py
EVO_WORKERS=2 setsid nohup ../.venv/bin/python bigeval.py exp/m3s2r1.py exp/echo2.py 999900 4 echo2_smoke > /root/carbon/data/echo2_smoke.out 2>&1 &
echo "LAUNCHED_PID=$!"
sleep 10
ps aux | grep echo2_smoke | grep -v grep | head -3
echo "--- E0 FIELD STILL ALIVE ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
echo "MARKER $MARK END"
