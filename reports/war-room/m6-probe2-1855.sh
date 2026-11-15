#!/bin/bash
# M6 probe 2 (staging + interface discovery): copy repo matches list into workdir,
# extract teacher name, print the small build scripts and the K4v2 build log header.
# Marker: M6PRB21855K3
MARK=M6PRB21855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB21855K3 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- REPO-LOG ---'
git -C /home/arm/junyang/carbon-repo log --oneline -5 2>&1
echo '--- WORKDIR-GIT ---'
git -C /home/arm/junyang/carbon rev-parse --short HEAD 2>&1; git -C /home/arm/junyang/carbon status --short 2>&1 | head -5
echo '--- REPO-MATCHES ---'
ls -l /home/arm/junyang/carbon-repo/data/m6_matches.json 2>&1
ls /home/arm/junyang/carbon/data/ | grep -i m6
echo '--- COPY-MATCHES ---'
cp /home/arm/junyang/carbon-repo/data/m6_matches.json /home/arm/junyang/carbon/data/m6_matches.json && echo COPY-OK
ls -l /home/arm/junyang/carbon/data/m6_matches.json
echo '--- MATCHES-SHAPE ---'
/home/arm/junyang/carbon/.venv/bin/python -c "
import json, collections
d=json.load(open('/home/arm/junyang/carbon/data/m6_matches.json'))
print('matches-type', type(d).__name__, 'len', len(d))
e=d[0]
print('first-entry', json.dumps(e, ensure_ascii=False)[:800])
names=collections.Counter()
for x in d:
    for side in ('a','b'):
        s=x.get('strategies',{}).get(side)
        if isinstance(s,dict) and s.get('name'): names[s['name']]+=1
        elif isinstance(s,str): names['FLAT:'+s]+=1
print('strategy-names', dict(names))
" 2>&1
echo '--- BUILD-TOP ---'
cat /home/arm/junyang/carbon/code/build_top.py 2>&1
echo '--- SET-UW ---'
cat /home/arm/junyang/carbon/code/set_uw.py 2>&1
echo '--- LADDER-CORPUS-FUNCS ---'
grep -n "^def \|^class \|^if __name__" /home/arm/junyang/carbon/code/ladder_corpus.py 2>&1
tail -40 /home/arm/junyang/carbon/code/ladder_corpus.py 2>&1
echo '--- K4V2-BUILD-LOG ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>&1 | head -15
for f in /home/arm/junyang/carbon/data/dq_logs/build_k4v2.log /home/arm/junyang/carbon/data/dq_logs/build_m7bsv2.log; do echo "== $f"; head -8 "$f" 2>/dev/null || echo NO-LOG; done
echo '--- M7BSV2-TRAIN-LOG-HEAD ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; head -4 "$f" 2>/dev/null || echo NO-LOG; done
echo '--- DS-META-SAMPLE ---'
/home/arm/junyang/carbon/.venv/bin/python -c "import json; d=json.load(open('/home/arm/junyang/carbon/data/ds_k4v2/meta.json')); print('ds_k4v2-meta-keys', sorted(d.keys())); print({k:(len(v) if isinstance(v,list) else v) for k,v in d.items()})" 2>&1
echo "M6PRB21855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
