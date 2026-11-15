#!/bin/bash
echo "M6CHAIN1907R5 TEACHER START $(date -u +%H:%M:%S)"
DGX="arm@10.211.43.155"
KEY="/root/.ssh/dgx_access"
timeout -k 30 240 ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$DGX" 'bash -s' <<'EOS'
cd /home/arm/junyang/carbon
echo "--- ds_m6 meta.json ---"
python3 -c "import json; m=json.load(open('data/ds_m6/meta.json')); print(json.dumps(m, indent=1)[:1500])"
echo "--- teacher names in matches ---"
python3 - <<'PYEOF'
import json, collections
d = json.load(open('data/m6_matches.json'))
print('n=', len(d))
print('name_a:', collections.Counter(m.get('name_a') for m in d))
print('name_b:', collections.Counter(m.get('name_b') for m in d))
PYEOF
echo "--- current epoch ---"
tail -1 data/weights/cnet2_m6r1_log.jsonl
tail -1 data/weights/cnet2_m6g1_log.jsonl
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps aux | grep cnet2_train | grep -v grep | awk '{print $2, $10}'
EOS
echo "SSH_RC=$?"
echo "M6CHAIN1907R5 TEACHER END"
