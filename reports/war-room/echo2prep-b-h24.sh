#!/bin/bash
# Runs on h24 via cloud assistant. READ ONLY: W5 weights + Echo2 submission pack provenance.
MARK="ECHO2PREPM3WH24"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
echo "--- M3W5 WEIGHTS ---"
ls -la /root/carbon/data/weights/ | grep -i m3w5
echo "--- M3W5 MD5 ---"
timeout 120 md5sum /root/carbon/data/weights/cnet2_m3w5*.npz 2>/dev/null
echo "--- NPZ SANITY (m3w5L) ---"
timeout 120 ../.venv/bin/python -c "
import numpy as np
d = np.load('/root/carbon/data/weights/cnet2_m3w5L.npz')
print('keys', len(d.files))
print('sample', d.files[:5])
" 2>&1 | head -5
cd /root/carbon/code
echo "--- SUBMISSION FILES ---"
ls -la exp/submission_echo2_w5q8.py exp/submission_echo2_w5f16.py exp/submission_echo1_q8nr.py 2>/dev/null
echo "--- W5Q8 HEAD ---"
timeout 30 head -40 exp/submission_echo2_w5q8.py
echo "--- W5Q8 SOURCE REFS ---"
timeout 30 grep -a -o 'cnet2_m3w5[A-Za-z0-9_.]*' exp/submission_echo2_w5q8.py | sort | uniq -c
timeout 30 grep -a -i -m3 'md5\|source\|from weights' exp/submission_echo2_w5q8.py | head -3
echo "--- M3S2R1 WRAPPER (h24) ---"
cat exp/m3s2r1.py
echo "--- M3S2R1 FINAL WEIGHTS MD5 ---"
timeout 120 md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz
ls -la /root/carbon/data/weights/cnet2_m3s2r1L.npz
echo "MARKER $MARK END"
