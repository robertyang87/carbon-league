#!/bin/bash
# Read-only: find the 2 games dropped at scan stage for ds_m6 (war-room 19:18 request: IDs for the record).
# Marker: M6DROP1918Q2
MARK=M6DROP1918Q2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6DROP1918Q2 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- M6-STAGE-LOGS ---'
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/ | grep -i 'm6' | head -10
echo '--- SCAN/DROP-LINES ---'
grep -h -i -E 'scan|drop|skip|orphan|exclud|invalid|empty' /home/arm/junyang/carbon/data/dq_logs/*m6*.log /home/arm/junyang/carbon/data/*m6*.log 2>/dev/null | head -30
echo '--- STAGE-LOG-FULL-TAIL ---'
for f in $(ls -t /home/arm/junyang/carbon/data/dq_logs/*m6*.log 2>/dev/null | head -2); do echo "== $f"; tail -15 "$f"; done
echo '--- M6-TRAIN-PROGRESS ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m6' | grep -v grep || echo NO-M6-TRAIN
tail -1 /home/arm/junyang/carbon/data/dq_logs/m6r1.log 2>/dev/null; tail -1 /home/arm/junyang/carbon/data/dq_logs/m6g1.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo "M6DROP1918Q2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
