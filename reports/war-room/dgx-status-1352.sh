#!/bin/bash
# DGX status follow-up probe, run 2026-11-12 ~13:52 CST
# Marker: DGXSTATUS1352M7QK (claim only output containing this marker)
MARK=DGXSTATUS1352M7QK
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1352M7QK DGX-BEGIN $(date '+%F %T %Z')"
hostname
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo '--- PROC-ALL ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m3s2' | grep -v grep
echo '--- WEIGHTS ---'
ls -lt --time-style=+%H:%M:%S /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -12
echo '--- CORPUS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'corpus_files_nonindex='
ls /home/arm/junyang/carbon/data/m3s2_corpus/ 2>/dev/null | grep -vcE 'index\.json|READY' || echo NA
echo '--- M3S2-SH ---'
grep -nE 'log|tee|npz|epoch|EPOCH|g1|r1|wait|&' /home/arm/junyang/carbon/code/m3s2.sh 2>/dev/null | head -40
echo '--- LOGS-RECENT ---'
find /home/arm/junyang/carbon -maxdepth 3 \( -name '*.log' -o -name '*.out' \) -newermt '2026-11-12 13:00' 2>/dev/null | head -12
echo '--- TAIL-NEWEST-LOGS ---'
for f in $(find /home/arm/junyang/carbon -maxdepth 3 \( -name '*.log' -o -name '*.out' \) -newermt '2026-11-12 13:00' 2>/dev/null | head -4); do echo "== $f"; tail -6 "$f"; done
echo "DGXSTATUS1352M7QK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
