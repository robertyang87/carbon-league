#!/bin/bash
# DGX status read-only probe, run 2026-11-12 ~13:47 CST (scheduled 13:43)
# Channel: cloud-assistant.yml -> L20 i-uf62k0o6zbe6j7zxnzg5 -> ssh DGX arm@10.211.43.155
# Marker: DGXSTATUS1343XK7Q (claim only output containing this marker)
MARK=DGXSTATUS1343XK7Q
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1343XK7Q DGX-BEGIN $(date '+%F %T %Z')"
hostname
uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC ---'
ps -eo pid,etime,args | grep -E 'm3s2|train|python' | grep -v grep | head -25
echo '--- WORKDIR ---'
W=$(find /home/arm -maxdepth 4 -name 'm3s2.sh' 2>/dev/null | head -1)
echo "m3s2.sh: $W"
D=$(dirname "$W" 2>/dev/null)
if [ -n "$D" ]; then cd "$D"; fi
pwd
echo '--- CORPUS ---'
python3 -c "import json;d=json.load(open('data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'corpus_files_nonindex='
ls data/m3s2_corpus/ 2>/dev/null | grep -vcE 'index\.json|READY' || echo NA
echo '--- TRAIN-LOGS ---'
ls -lt *.log 2>/dev/null | head -8
for f in $(ls -t *.log 2>/dev/null | head -4); do echo "== $f"; tail -4 "$f"; done
echo '--- CKPT ---'
find . -maxdepth 3 \( -iname '*ckpt*' -o -iname '*ep1[0-9]*' \) -newermt '2026-11-12 11:00' 2>/dev/null | head -15
ls -lt --time-style=+%H:%M *ckpt* */*ckpt* 2>/dev/null | head -10
echo '--- M7BS ---'
ls -d *m7bs* *M7Bs* 2>/dev/null; find . -maxdepth 3 -iname '*m7bs*' 2>/dev/null | head -10
echo "DGXSTATUS1343XK7Q DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
