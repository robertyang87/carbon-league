#!/bin/bash
# M6 probe (read-only + git pull): confirm DGX state, corpus counts, teacher name,
# and the exact CLI usage of the build/train entry points, before launching the chain.
# Marker: M6PROBE1855Q7X1
MARK=M6PROBE1855Q7X1
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PROBE1855Q7X1 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- REPO-HEAD ---'
timeout 60 git -C /home/arm/junyang/carbon-repo pull --ff-only 2>&1 | tail -3
git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD 2>/dev/null; git -C /home/arm/junyang/carbon-repo branch --show-current 2>/dev/null
echo '--- WORKDIR-LAYOUT ---'
ls /home/arm/junyang/carbon/ | head -20
ls /home/arm/junyang/carbon/code/ 2>/dev/null | grep -E 'ladder_corpus|build_top|set_uw|cnet2_train|cnet2_build' || echo NO-CODE-MATCH
echo '--- TRAIN-PROC + GPU ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- M6-COUNTS ---'
ls /home/arm/junyang/carbon/data/m6_corpus/*.json.gz 2>/dev/null | wc -l
ls /home/arm/junyang/carbon/data/m6_corpus/ | head -5
ls /home/arm/junyang/carbon/data/m6_corpus/index.json 2>/dev/null || echo NO-WORKDIR-INDEX
ls -d /home/arm/junyang/carbon/data/m6_corpus/replays 2>/dev/null || echo NO-REPLAYS-SUBDIR
/home/arm/junyang/carbon/.venv/bin/python -c "import json; d=json.load(open('/home/arm/junyang/carbon/data/m6_matches.json')); print('matches-len', len(d) if isinstance(d,list) else list(d.keys())[:5])" 2>&1
/home/arm/junyang/carbon/.venv/bin/python -c "
import json
d=json.load(open('/home/arm/junyang/carbon/data/m6_matches.json'))
e=d[0] if isinstance(d,list) else d
print('first-entry-keys', sorted(e.keys()))
print('first-entry', json.dumps(e, ensure_ascii=False)[:600])
" 2>&1
echo '--- TEACHER-NAMES ---'
/home/arm/junyang/carbon/.venv/bin/python -c "
import json, collections
d=json.load(open('/home/arm/junyang/carbon/data/m6_matches.json'))
names=collections.Counter()
for e in (d if isinstance(d,list) else []):
    for side in ('a','b'):
        s=e.get('strategies',{}).get(side)
        if isinstance(s,dict) and s.get('name'): names[s['name']]+=1
print('strategy-names', dict(names))
" 2>&1
echo '--- ENTRYPOINT-USAGE ---'
cd /home/arm/junyang/carbon/code 2>/dev/null && pwd
for f in ladder_corpus.py build_top.py set_uw.py cnet2_train.py; do echo "== $f"; ls -l "$f" 2>/dev/null || echo MISSING; done
echo '--- LADDER-CORPUS-HELP ---'
/home/arm/junyang/carbon/.venv/bin/python ladder_corpus.py --help 2>&1 | head -30 || true
echo '--- TRAIN-HELP ---'
/home/arm/junyang/carbon/.venv/bin/python cnet2_train.py --help 2>&1 | head -40 || true
echo '--- BC-INIT-WEIGHTS ---'
ls -l /home/arm/junyang/carbon/data/weights/cnet2_arch_n1_last.npz /home/arm/junyang/carbon/data/weights/cnet2_b128m2t3.npz 2>&1
ls /home/arm/junyang/carbon/data/ | grep '^ds_' | head -10
echo "M6PROBE1855Q7X1 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
