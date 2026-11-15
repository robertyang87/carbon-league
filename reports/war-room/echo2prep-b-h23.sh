#!/bin/bash
# Runs on h23 via cloud assistant. READ ONLY: W5 weights provenance on h23.
MARK="ECHO2PREPM3WH23"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
echo "--- M3W5 WEIGHTS ---"
ls -la /root/carbon/data/weights/ | grep -i m3w5
echo "--- M3W5 MD5 ---"
timeout 120 md5sum /root/carbon/data/weights/cnet2_m3w5*.npz 2>/dev/null
echo "--- NPZ SANITY (m3w5L) ---"
cd /root/carbon/code
timeout 120 ../.venv/bin/python -c "
import numpy as np
d = np.load('/root/carbon/data/weights/cnet2_m3w5L.npz')
print('keys', len(d.files))
print('sample', d.files[:5])
" 2>&1 | head -5
echo "--- E0PRE STILL ALIVE ---"
ps aux | grep 'm3s2r1pre_e0' | grep -v grep | head -2
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl
echo "MARKER $MARK END"
