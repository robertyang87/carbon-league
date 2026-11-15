#!/bin/bash
# Read-only follow-up: full stage/build logs + diff index.json (191) vs ds_m6 accepted games to name the 2 dropped.
# Marker: M6DROP1918B2
MARK=M6DROP1918B2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6DROP1918B2 DGX-BEGIN $(date '+%F %T %Z')"
D=/home/arm/junyang/carbon/data
echo '--- FULL m6_stage.log ---'
cat $D/dq_logs/m6_stage.log
echo '--- FULL build_m6.log ---'
cat $D/dq_logs/build_m6.log
echo '--- DIFF index vs ds_m6 ---'
python3 - <<'PYEOF'
import json
try:
    idx = json.load(open('/home/arm/junyang/carbon/data/m6_corpus/index.json'))
    print('index type:', type(idx).__name__, 'len:', len(idx))
    if isinstance(idx, list) and idx:
        print('index[0] keys:', sorted(idx[0].keys()))
        ikeys = set()
        for e in idx:
            ikeys.add(e.get('id') or e.get('game') or str(e))
    elif isinstance(idx, dict):
        print('index dict keys sample:', list(idx.keys())[:5])
        ikeys = set(idx.keys())
    else:
        ikeys = set()
except Exception as e:
    print('index load error:', e); ikeys = set()
try:
    meta = json.load(open('/home/arm/junyang/carbon/data/ds_m6/meta.json'))
    print('meta keys:', sorted(meta.keys()))
    print('meta games field:', meta.get('games') if isinstance(meta.get('games'), int) else (len(meta['games']) if isinstance(meta.get('games'), list) else meta.get('games')))
    if isinstance(meta.get('games'), list):
        dkeys = set(g.get('id') or g.get('game') or str(g) for g in meta['games'])
        print('dropped (in index, not in ds):', sorted(ikeys - dkeys))
        print('extra (in ds, not in index):', sorted(dkeys - ikeys))
    else:
        print('ds game list not in meta; check shards dir')
        import os
        print('ds_m6 files:', sorted(os.listdir('/home/arm/junyang/carbon/data/ds_m6'))[:20])
except Exception as e:
    print('meta load error:', e)
PYEOF
echo "M6DROP1918B2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
