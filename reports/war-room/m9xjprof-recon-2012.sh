#!/bin/bash
# M9XJ it260 teacher profile RECON (stage 1): repo pull, corpus/matches counts, schema dump of one match entry + one replay.
# CPU-only, read-only apart from git pull --ff-only. Marker: M9XJPROF2012R1
MARK=M9XJPROF2012R1
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M9XJPROF2012R1 DGX-BEGIN $(date '+%F %T %Z')"
hostname
cd /home/arm/junyang/carbon-repo || { echo NO-REPO; exit 1; }
echo "HEAD-BEFORE $(git rev-parse HEAD)"
timeout 60 git pull --ff-only 2>&1 | tail -2
echo "HEAD-AFTER $(git rev-parse HEAD)"
echo "BRANCH $(git rev-parse --abbrev-ref HEAD)"
echo -n "corpus-gz-count: "; ls data/m9xj_corpus/*.json.gz 2>/dev/null | wc -l
ls data/m9xj_corpus/ | head -4
ls -la data/m9xj_matches.json
python3 - <<'PYEOF'
import json, gzip, glob
m = json.load(open('data/m9xj_matches.json'))
print('matches type', type(m).__name__, 'len', len(m) if hasattr(m, '__len__') else '?')
if isinstance(m, dict):
    print('matches top keys', list(m.keys())[:12])
    for k, v in m.items():
        if isinstance(v, list) and v:
            print('list under key', k, 'len', len(v)); m = v; break
e = m[0]
print('entry0:', json.dumps(e, ensure_ascii=False)[:1500])
e2 = m[len(m)//2]
print('entryMid keys:', sorted(e2.keys()))
print('entryMid:', json.dumps(e2, ensure_ascii=False)[:600])
files = sorted(glob.glob('data/m9xj_corpus/*.json.gz'))
print('n files', len(files), 'first', files[0])
d = json.load(gzip.open(files[0]))
print('replay top keys', list(d.keys()))
st = d['steps']
print('n step entries', len(st), 'st0 type', type(st[0]).__name__, 'st0 len', len(st[0]) if hasattr(st[0], '__len__') else '?')
e0 = st[0][0]
print('st[0][0] keys', list(e0.keys()) if isinstance(e0, dict) else type(e0).__name__)
obs = e0['observation']
print('obs keys', list(obs.keys()))
pl = obs['players']
print('players type', type(pl).__name__, 'len', len(pl))
p0 = pl[0]
print('player comp types:', [(i, type(c).__name__, (len(c) if hasattr(c, '__len__') else c)) for i, c in enumerate(p0)])
t3 = p0[3]
if isinstance(t3, dict):
    print('trees sample items:', list(t3.items())[:3])
print('tree counts t0 p0/p1:', len(pl[0][3]), len(pl[1][3]))
mid = st[len(st)//2][0]['observation']['players']
last = st[-1][0]['observation']['players']
print('tree counts mid p0/p1:', len(mid[0][3]), len(mid[1][3]))
print('tree counts last p0/p1:', len(last[0][3]), len(last[1][3]))
print('obs step t0:', obs.get('step'), 'last:', st[-1][0]['observation'].get('step'))
try:
    o1 = st[len(st)//2][1]['observation']
    print('entry1 mid players[1][3] count:', len(o1['players'][1][3]), 'vs entry0:', len(mid[1][3]))
    print('entry1 mid players[0][3] count:', len(o1['players'][0][3]), 'vs entry0:', len(mid[0][3]))
except Exception as ex:
    print('entry1 check err', repr(ex)[:200])
for t in (1, 2):
    try:
        print('action t', t, 'p0:', json.dumps(st[t][0].get('action'))[:160], ' p1:', json.dumps(st[t][1].get('action'))[:160])
    except Exception as ex:
        print('action err', repr(ex)[:150])
PYEOF
echo "M9XJPROF2012R1 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
