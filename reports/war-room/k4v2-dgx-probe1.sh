#!/bin/bash
MARK=K4DGXPRB1
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "K4DGXPRB1 DGX-BEGIN $(date '+%F %T %Z')"
hostname
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- M7BSV2-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; ls -l --time-style=+%F_%T "$f"; tail -2 "$f"; done
echo '--- REPO ---'
cd /home/arm/junyang/carbon-repo && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD
timeout 60 git ls-remote origin cnet2-search-v12 2>&1 | head -2
echo '--- K4V2-REPO-DATA ---'
ls data/k4v2_corpus 2>/dev/null | wc -l; ls data/k4v2_corpus 2>/dev/null | head -6
python3 -c "import json;d=json.load(open('data/k4v2_matches.json'));print('matches',len(d));import pprint;pprint.pprint(d[0]); print('names', sorted(set([x.get('name_a','') for x in d]+[x.get('name_b','') for x in d])))" 2>&1 | head -30
echo '--- K4V2-LOCAL-DATA ---'
ls /home/arm/junyang/carbon/data/ | grep -i k4v2 || echo NO-K4V2-LOCAL
ls /home/arm/junyang/carbon/data/k4v2_corpus 2>/dev/null || echo NO-K4V2-CORPUS-LOCAL
ls /home/arm/junyang/carbon/data/ds_k4v2 2>/dev/null || echo NO-DS-K4V2
echo '--- EXISTING-CORPUS-SHAPE ---'
python3 - <<'PYEOF'
import json
for name in ('m3s2','m7bs','carago','fable'):
    try:
        d=json.load(open(f'/home/arm/junyang/carbon/data/{name}_corpus/index.json'))
        print(name, 'index entries', len(d), 'first strategies:', json.dumps(d[0]['strategies'], ensure_ascii=False)[:300], 'keys:', sorted(d[0].keys()))
    except Exception as e:
        print(name, 'ERR', e)
PYEOF
echo '--- CODE ---'
ls /home/arm/junyang/carbon/code/ | grep -E 'ladder_corpus|build_top|set_uw|cnet2_build'
echo "K4DGXPRB1 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
