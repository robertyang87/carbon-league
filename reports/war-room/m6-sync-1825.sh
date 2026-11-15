#!/bin/bash
# m6_corpus alignment sync (war-room instruction 18:24, executor named: DGX reporter).
# Read + sync files only. Must NOT touch running m7bsv2 training processes.
# Marker: M6SYNC1825XQ7
MARK=M6SYNC1825XQ7
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6SYNC1825XQ7 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- TRAIN-STILL-RUNNING-CHECK (before) ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN
echo '--- REPO-PULL ---'
cd /home/arm/junyang/carbon-repo
timeout 60 git pull --ff-only 2>&1 | tail -3
echo "repo HEAD: $(git rev-parse HEAD) $(git log -1 --format='%ci %s')"
echo '--- REPO-COUNTS ---'
echo -n "repo m6_corpus replays: "; ls data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "repo m6_corpus total files: "; find data/m6_corpus -type f | wc -l
echo -n "repo m6_matches.json entries: "; python3 -c "import json;print(len(json.load(open('data/m6_matches.json'))))" 2>/dev/null || echo NO-MATCHES-FILE
echo -n "repo m6_corpus index.json entries: "; python3 -c "import json;d=json.load(open('data/m6_corpus/index.json'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo NO-INDEX-FILE
ls -l --time-style=+%F_%H:%M:%S data/m6_corpus/index.json 2>/dev/null || echo NO-INDEX-LS
echo '--- WORKDIR-COUNTS-BEFORE ---'
echo -n "workdir m6_corpus replays: "; ls /home/arm/junyang/carbon/data/m6_corpus/*.json.gz 2>/dev/null | wc -l
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m6_corpus/index.json 2>/dev/null || echo NO-INDEX-LS-WORK
echo '--- SYNC (repo authoritative) ---'
rsync -a /home/arm/junyang/carbon-repo/data/m6_corpus/ /home/arm/junyang/carbon/data/m6_corpus/ 2>&1 | tail -2
echo "rsync rc=$?"
echo '--- WORKDIR-COUNTS-AFTER ---'
echo -n "workdir m6_corpus replays after: "; ls /home/arm/junyang/carbon/data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "workdir m6_corpus total files after: "; find /home/arm/junyang/carbon/data/m6_corpus -type f | wc -l
echo -n "workdir m6_corpus index.json entries after: "; python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m6_corpus/index.json'));print(len(d) if isinstance(d,list) else len(d.get('entries',d)))" 2>/dev/null || echo NO-INDEX-FILE-AFTER
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m6_corpus/index.json 2>/dev/null || echo NO-INDEX-LS-AFTER
echo '--- TRAIN-STILL-RUNNING-CHECK (after) ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN-AFTER
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo "M6SYNC1825XQ7 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
