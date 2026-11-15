#!/bin/bash
# DGX status probe 01:59 CST follow-up, marker DGXSTATUS0158B
set -u
M=DGXSTATUS0158B
echo "=== $M start $(date '+%F %T %Z') ==="
echo "--- k0_calib tree ---"
ls -la /home/arm/junyang/k0_calib/ 2>&1 | head -20
echo "--- duels dir ---"
ls -la /home/arm/junyang/k0_calib/duels/ 2>&1 | head -20
echo "--- cells dir ---"
ls -la /home/arm/junyang/k0_calib/cells/ 2>&1 | head -20
echo "--- duel out file check ---"
ls -la /home/arm/junyang/k0_calib/duels/lr0.0002_s100_b1_seed7.json 2>&1
echo "--- ds counts ---"
python3 -c "
import json
for d in ('ds_carago','ds_fable'):
    m=json.load(open(f'/home/arm/junyang/carbon/data/{d}/meta.json'))
    print(d, 'keys=', list(m.keys())[:10])
    # try common count fields
    for k in ('n_samples','samples','total','count','num_samples'):
        if k in m: print(' ', k, '=', m[k])
    if 'games' in m: print('  games len=', len(m['games']))
" 2>&1
echo "--- krypton locations ---"
ls /home/arm/junyang/carbon-repo/data/ 2>&1
find /home/arm/junyang/carbon-repo -maxdepth 3 -name '*krypton*' 2>/dev/null | head
find /home/arm/junyang/carbon -maxdepth 3 -name '*krypton*' 2>/dev/null | head
echo "=== $M end ==="
