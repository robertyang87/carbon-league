#!/bin/bash
echo "M6CHAIN1907R3 PROBE START $(date -u +%H:%M:%S)"
DGX="arm@10.211.43.155"
KEY="/root/.ssh/dgx_access"
timeout -k 30 240 ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$DGX" 'bash -s' <<'EOS'
cd /home/arm/junyang/carbon
echo "--- workdir m6_corpus entries ---"
ls -la data/m6_corpus/ | head -12
find data/m6_corpus -type f | wc -l
echo "--- ds_m6 stat ---"
ls -la data/ds_m6/ | head -20
find data/ds_m6 -type f | wc -l
stat -c '%y %n' data/ds_m6/* 2>/dev/null | head -20
echo "--- ds_m6 sample/game count ---"
python3 - <<'PYEOF'
import numpy as np, json, os, glob
d = 'data/ds_m6'
for f in sorted(glob.glob(d + '/*')):
    print(f, os.path.getsize(f))
try:
    a = np.load(d + '/hist_move.npy')
    print('hist_move shape:', a.shape)
except Exception as e:
    print('hist_move load err:', e)
for name in ['index.json', 'manifest.json', 'meta.json', 'READY']:
    p = os.path.join(d, name)
    if os.path.exists(p):
        print(name, open(p).read()[:500])
PYEOF
echo "--- train logs modified today ---"
find /home/arm/junyang/carbon -maxdepth 3 -name '*.log' -newermt '2026-11-12 18:30' -exec ls -la {} \; 2>/dev/null | head -20
ls -la data/weights/ 2>/dev/null | grep -i m6
echo "--- train proc full cmdline + log tails ---"
ps aux | grep cnet2_train | grep -v grep
for f in m6r1 m6g1; do echo "== $f =="; ls -la data/weights/cnet2_${f}.npz 2>/dev/null || echo "no weight yet"; done
ls -lat *.log *.out nohup* 2>/dev/null | head
EOS
echo "SSH_RC=$?"
echo "M6CHAIN1907R3 PROBE END"
