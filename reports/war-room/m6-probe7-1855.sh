#!/bin/bash
# M6 probe 7: does cnet2_build.scan use the strategies id? Is M6's strategy id
# in any leaderboard.json snapshot on DGX? What does ladder.py expose?
# Marker: M6PRB71855K3
MARK=M6PRB71855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB71855K3 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- SCAN-ID-USE ---'
grep -n "strategies" /home/arm/junyang/carbon/code/cnet2_build.py 2>&1 | head -20
echo '--- LEADERBOARD-SNAPSHOTS ---'
ls /home/arm/junyang/carbon/data/*/leaderboard.json 2>&1
/home/arm/junyang/carbon/.venv/bin/python -c "
import json, glob
for p in sorted(glob.glob('/home/arm/junyang/carbon/data/*/leaderboard.json')):
    try:
        d=json.load(open(p))
    except Exception as e:
        print(p, 'LOAD-FAIL', e); continue
    rows=d if isinstance(d,list) else d.get('strategies', d.get('rows', []))
    print(p, 'type', type(d).__name__, 'rows', len(rows) if isinstance(rows,list) else '?')
    for r in (rows if isinstance(rows,list) else []):
        if isinstance(r,dict) and 'RyanX-AIPowercore-M6' in str(r.get('name','')):
            print('  M6-ENTRY', json.dumps(r, ensure_ascii=False)[:300])
" 2>&1
echo '--- M6-IN-OTHER-INDICES ---'
grep -o '"name": "RyanX-AIPowercore-M6"[^}]*}' /home/arm/junyang/carbon/data/m7bs_corpus/index.json 2>/dev/null | head -2
grep -o '{\s*"id": "[^"]*", "name": "RyanX-AIPowercore-M6"' /home/arm/junyang/carbon/data/*/index.json 2>/dev/null | head -5
echo '--- LADDER-API ---'
ls /home/arm/junyang/carbon/code/ladder.py 2>&1
grep -n "^def \|^class " /home/arm/junyang/carbon/code/ladder.py 2>&1 | head -30
echo "M6PRB71855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
