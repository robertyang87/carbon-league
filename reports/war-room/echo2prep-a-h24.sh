#!/bin/bash
# Runs on h24 via cloud assistant. READ ONLY: E0 field aliveness + Echo2 artifact hunt.
MARK="ECHO2PREPK7QX24"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
echo "--- BIGEVAL PROCS ---"
ps aux | grep -i bigeval | grep -v grep
echo "--- E0 LIVE COUNT 1 ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
sleep 25
echo "--- E0 LIVE COUNT 2 (+25s) ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
timeout 30 tail -1 /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
echo "--- FIND ECHO ---"
timeout 60 find /root/carbon -iname '*echo*' 2>/dev/null
echo "--- EXP LIST ---"
ls /root/carbon/code/exp/
echo "--- ECHO_E0.PY ---"
cat /root/carbon/code/exp/echo_e0.py
echo "--- WEIGHTS LS ---"
ls -la /root/carbon/data/weights/ | head -60
echo "--- PACKS LS ---"
ls -la /root/carbon/data/packs/ 2>/dev/null
echo "--- ECHO MD5 ---"
timeout 60 find /root/carbon -iname '*echo*' -type f -exec md5sum {} \; 2>/dev/null
echo "MARKER $MARK END"
