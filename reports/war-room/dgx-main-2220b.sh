#!/bin/bash
# Characterize the one untracked file blocking the DGX main checkout. Read-only.
# Marker: DGXMAIN2220B
MARK=DGXMAIN2220B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN2220B DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN2220B NO-REPO"; exit 1; }
F=code/matches_d679ed45ab595e8b7bb1ea51c3451154.json
echo '--- FILE ---'
ls -l "$F"
stat -c 'mtime=%y size=%s' "$F"
echo '--- HEAD ---'
head -c 600 "$F"; echo
echo '--- MD5-AND-COMPARE ---'
python3 - <<'PYEOF'
import json, hashlib, os
f = 'code/matches_d679ed45ab595e8b7bb1ea51c3451154.json'
raw = open(f,'rb').read()
print('md5=', hashlib.md5(raw).hexdigest(), 'bytes=', len(raw))
try:
    d = json.loads(raw)
    print('type=', type(d).__name__, 'len=', len(d) if hasattr(d,'__len__') else 'n/a')
    if isinstance(d, list) and d:
        print('first keys=', list(d[0].keys())[:10] if isinstance(d[0], dict) else d[0])
    elif isinstance(d, dict):
        print('dict keys sample=', list(d.keys())[:10])
except Exception as e:
    print('json parse ERR', e)
for cand in ['data/m9xj320_matches.json', 'data/m9xj_matches.json', 'data/m3s2_matches.json']:
    if os.path.exists(cand):
        b = open(cand,'rb').read()
        print(cand, 'bytes=', len(b), 'md5=', hashlib.md5(b).hexdigest(), 'SAME' if b == raw else 'diff')
    else:
        print(cand, 'MISSING')
PYEOF
echo '--- STATUS-AGAIN ---'
git status --porcelain
echo '--- REMOTE-MAIN-TIP ---'
git ls-remote origin main
echo '--- LOCAL-BRANCHES ---'
git branch -vv | head -10
echo "DGXMAIN2220B DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
