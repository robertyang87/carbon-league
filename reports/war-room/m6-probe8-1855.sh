#!/bin/bash
# M6 probe 8: raw-text search for M6 in the leaderboard snapshots (any id field?),
# plus first-row keys and file ages. Last TID check before building the chain.
# Marker: M6PRB81855K3
MARK=M6PRB81855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB81855K3 DGX-BEGIN $(date '+%F %T %Z')"
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/m2_corpus/leaderboard.json /home/arm/junyang/carbon/data/m2s_corpus/leaderboard.json 2>&1
for f in /home/arm/junyang/carbon/data/m2_corpus/leaderboard.json /home/arm/junyang/carbon/data/m2s_corpus/leaderboard.json; do echo "== $f"; grep -c "RyanX-AIPowercore-M6" "$f" 2>&1; grep -o '.\{0,60\}RyanX-AIPowercore-M6.\{0,100\}' "$f" 2>/dev/null | head -2 | cut -c1-300; done
/home/arm/junyang/carbon/.venv/bin/python -c "
import json
d=json.load(open('/home/arm/junyang/carbon/data/m2_corpus/leaderboard.json'))
rows=d if isinstance(d,list) else d.get('strategies', d.get('rows', d.get('data', [])))
print('top-keys', sorted(d.keys()) if isinstance(d,dict) else 'list')
print('first-row-keys', sorted(rows[0].keys()) if rows and isinstance(rows[0],dict) else 'none')
print('first-row', json.dumps(rows[0], ensure_ascii=False)[:300] if rows else 'empty')
" 2>&1
echo "M6PRB81855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
