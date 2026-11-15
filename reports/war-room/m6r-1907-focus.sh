#!/bin/bash
echo "M6CHAIN1907R4 FOCUS START $(date -u +%H:%M:%S)"
DGX="arm@10.211.43.155"
KEY="/root/.ssh/dgx_access"
timeout -k 30 240 ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$DGX" 'bash -s' <<'EOS'
cd /home/arm/junyang/carbon
echo "--- workdir m6_corpus ---"
ls data/m6_corpus/ | head -5
find data/m6_corpus -type f | wc -l
find data/m6_corpus -type d | head
echo "--- ds_m6 mtimes ---"
stat -c '%y %s %n' data/ds_m6/* | head -20
echo "--- ds_m6 shapes ---"
python3 - <<'PYEOF'
import numpy as np, os
for f in ['hist_move.npy', 'hist_cell.npy', 'hist_cargo.npy', 'carbon.npy', 'gid.npy']:
    p = 'data/ds_m6/' + f
    if os.path.exists(p):
        a = np.load(p)
        print(f, a.shape, a.dtype)
PYEOF
echo "--- teacher name ---"
python3 -c "import json; d=json.load(open('data/m6_matches.json')); print('n=', len(d)); print('first keys:', list(d[0].keys()) if isinstance(d, list) else list(d.keys())[:10]); import collections; print(collections.Counter([ (m.get('a') if isinstance(m, dict) else None) for m in (d if isinstance(d, list) else d.get('matches', [])) ]))" 2>&1 | head -10
echo "--- ep0 vals ---"
cat data/weights/cnet2_m6r1_log.jsonl 2>/dev/null | head -3
cat data/weights/cnet2_m6g1_log.jsonl 2>/dev/null | head -3
echo "--- gpu + procs ---"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps aux | grep cnet2_train | grep -v grep | awk '{print $2, $10, $11, $12, $13, $14}'
EOS
echo "SSH_RC=$?"
echo "M6CHAIN1907R4 FOCUS END"
