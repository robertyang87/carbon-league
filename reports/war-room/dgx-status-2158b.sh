#!/bin/bash
# DGX micro-probe 2158b: Echo2 count in frozen 91 (field-name-robust), manifest/READY
# contents, chain-log whereabouts, train-log location. Read-only. Marker: DGXSTATUS2158E2
MARK=DGXSTATUS2158E2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2158E2 DGX-BEGIN $(date '+%F %T %Z')"
python3 -c "
import json
for path in ('/home/arm/junyang/carbon/data/m9xj320_corpus/index.json','/home/arm/junyang/carbon/data/m9xj320_matches.json'):
    d=json.load(open(path))
    e=d if isinstance(d,list) else d.get('entries',d)
    print(path,'entries=',len(e))
    print(' first-entry keys=',sorted(e[0].keys()))
    print(' first-entry sample=',json.dumps(e[0],ensure_ascii=False)[:400])
    n=[x for x in e if 'echo2' in json.dumps(x).lower()]
    print(' entries containing echo2=',len(n))
    for x in n:
        print('  ',json.dumps(x,ensure_ascii=False)[:300])
" 2>&1
echo '--- MD5-MATCHES ---'
md5sum /home/arm/junyang/carbon/data/m9xj320_matches.json
echo '--- MANIFEST+READY ---'
cat /home/arm/junyang/carbon/data/m9xj320_manifest.json
cat /home/arm/junyang/carbon/data/M9XJ320_READY.txt
cat /home/arm/junyang/carbon/data/m9xj320_chain.state
echo '--- DQ-LOGS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/dq_logs/ | head -10
echo '--- CHAIN-LOG-WHEREABOUTS ---'
timeout 30 find /home/arm/junyang/carbon -maxdepth 4 \( -name 'm320n_chain_outer.log' -o -name 'build_m9xj320.log' -o -name 'replica_chain_m9xj320.log' -o -name '*m9xj320*.log' \) 2>/dev/null | head -20
echo '--- TRAIN-LOG-CANDIDATES ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/*.log /home/arm/junyang/carbon/*.log 2>/dev/null | head -15
echo '--- TRAIN-EPOCH-SNAPSHOT ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj320*.npz | head -6
ps -eo pid,etime,args | grep cnet2_train | grep -v grep
echo "DGXSTATUS2158E2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
