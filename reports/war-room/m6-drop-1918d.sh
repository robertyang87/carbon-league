#!/bin/bash
# Read-only: check the 2 scan-dropped M6 games (IDs from verification agent run 43208172658): bad files or filter caliber?
# Marker: M6DROP1918D4
MARK=M6DROP1918D4
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6DROP1918D4 DGX-BEGIN $(date '+%F %T %Z')"
python3 - <<'PYEOF'
import json, gzip, os
R = '/home/arm/junyang/carbon/data/m6_corpus/replays/'
idx = json.load(open('/home/arm/junyang/carbon/data/m6_corpus/index.json'))
want = {'1b375bb6a65d1d7ecbc7014b78806757', '27fbfcb00f613d823e231a32cf95debc'}
for e in idx:
    if e.get('id') in want:
        f = e['file']
        print('ID:', e['id'])
        print('  index: file=%s game=%s steps=%s winner=%s scores=%s' % (f, e.get('game'), e.get('steps'), e.get('winner'), e.get('scores')))
        print('  strategies:', e.get('strategies'))
        p = R + os.path.basename(f)
        print('  replay path:', p, 'exists:', os.path.exists(p), 'size:', os.path.getsize(p) if os.path.exists(p) else None)
        try:
            with gzip.open(p, 'rt') as fh:
                r = json.load(fh)
            st = r.get('steps')
            print('  replay parses OK; steps len:', len(st) if isinstance(st, list) else st)
        except Exception as ex:
            print('  replay READ ERROR:', type(ex).__name__, str(ex)[:200])
PYEOF
echo "M6DROP1918D4 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
