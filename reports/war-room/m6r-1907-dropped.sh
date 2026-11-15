#!/bin/bash
echo "M6CHAIN1907R6 DROPPED START $(date -u +%H:%M:%S)"
DGX="arm@10.211.43.155"
KEY="/root/.ssh/dgx_access"
timeout -k 30 240 ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$DGX" 'bash -s' <<'EOS'
cd /home/arm/junyang/carbon
python3 - <<'PYEOF'
import json
meta = json.load(open('data/ds_m6/meta.json'))
games = meta['games']
print('meta games:', len(games))
have = set()
for g in games:
    k = g.get('key', '')
    have.add(k)
    if k.endswith('-1.json.gz'):
        have.add(k[:-len('-1.json.gz')])
d = json.load(open('data/m6_matches.json'))
print('matches:', len(d))
missing = []
for m in d:
    mid = m.get('id', '')
    if mid not in have:
        missing.append(m)
print('missing count:', len(missing))
for m in missing:
    print('MISSING:', m.get('id'), '|', m.get('name_a'), 'vs', m.get('name_b'), '| scores', m.get('scores'), '| winner', m.get('winner'))
PYEOF
echo "--- current epoch ---"
tail -1 data/weights/cnet2_m6r1_log.jsonl
tail -1 data/weights/cnet2_m6g1_log.jsonl
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps aux | grep cnet2_train | grep -v grep | awk '{print $2, $10}'
EOS
echo "SSH_RC=$?"
echo "M6CHAIN1907R6 DROPPED END"
