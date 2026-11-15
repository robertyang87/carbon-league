#!/bin/bash
# Read-only: correct diff (index 'file' vs ds 'key') to name the 2 scan-dropped games + inspect their replays.
# Marker: M6DROP1918C3
MARK=M6DROP1918C3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6DROP1918C3 DGX-BEGIN $(date '+%F %T %Z')"
python3 - <<'PYEOF'
import json, gzip, os
idx = json.load(open('/home/arm/junyang/carbon/data/m6_corpus/index.json'))
meta = json.load(open('/home/arm/junyang/carbon/data/ds_m6/meta.json'))
games = meta['games']
print('index entries:', len(idx), 'ds games:', len(games))
idx_by_file = {e['file']: e for e in idx}
ds_keys = set(g['key'] for g in games)
dropped = sorted(set(idx_by_file) - ds_keys)
print('DROPPED count:', len(dropped))
for f in dropped:
    e = idx_by_file[f]
    print('DROPPED file:', f)
    print('  index entry: game=%s steps=%s winner=%s scores=%s strategies=%s' % (e.get('game'), e.get('steps'), e.get('winner'), e.get('scores'), e.get('strategies')))
    path = '/home/arm/junyang/carbon/data/m6_corpus/replays/' + f
    print('  replay exists:', os.path.exists(path), 'size:', os.path.getsize(path) if os.path.exists(path) else None)
    try:
        with gzip.open(path, 'rt') as fh:
            r = json.load(fh)
        print('  replay parses OK; top keys:', sorted(r.keys())[:10])
        steps = r.get('steps')
        print('  replay steps len:', len(steps) if isinstance(steps, list) else steps)
        if isinstance(steps, list) and steps:
            print('  first step keys:', sorted(steps[0].keys()) if isinstance(steps[0], dict) else type(steps[0]))
    except Exception as ex:
        print('  replay READ ERROR:', type(ex).__name__, str(ex)[:200])
PYEOF
echo '--- M6-TRAIN-PROGRESS ---'
tail -1 /home/arm/junyang/carbon/data/dq_logs/m6r1.log 2>/dev/null; tail -1 /home/arm/junyang/carbon/data/dq_logs/m6g1.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo "M6DROP1918C3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
