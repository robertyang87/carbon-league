#!/bin/bash
# Runs on h23 via cloud assistant. READ ONLY: E0 pre-field aliveness + Echo2 artifact hunt.
MARK="ECHO2PREPK7QX23"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
echo "--- BIGEVAL PROCS ---"
ps aux | grep -i bigeval | grep -v grep
echo "--- E0PRE LIVE COUNT 1 ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl
sleep 25
echo "--- E0PRE LIVE COUNT 2 (+25s) ---"
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl
timeout 30 tail -1 /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl
echo "--- FIND ECHO ---"
timeout 60 find /root/carbon -iname '*echo*' 2>/dev/null
echo "--- EXP LIST ---"
ls /root/carbon/code/exp/
echo "--- ECHO_E0.PY (if present) ---"
cat /root/carbon/code/exp/echo_e0.py 2>/dev/null || echo NO_ECHO_E0_ON_H23
echo "--- WEIGHTS LS ---"
ls -la /root/carbon/data/weights/ | head -60
echo "--- PACKS LS ---"
ls -la /root/carbon/data/packs/ 2>/dev/null
echo "--- ECHO MD5 ---"
timeout 60 find /root/carbon -iname '*echo*' -type f -exec md5sum {} \; 2>/dev/null
echo "--- M3S2R1 FINAL WEIGHTS ON H23? ---"
ls -la /root/carbon/data/weights/cnet2_m3s2r1L.npz 2>/dev/null || echo NO_FINAL_R1_ON_H23
ls -la /root/carbon/code/exp/m3s2r1.py 2>/dev/null || echo NO_M3S2R1_WRAPPER_ON_H23
ls -la /root/carbon/code/exp/m3s2r1pre.py 2>/dev/null
echo "MARKER $MARK END"
