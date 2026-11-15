#!/bin/bash
# DGX status probe 6 (manual, user-triggered): cross-check war-room M7Bs fix/restart.
# War-room dispatched 定性＋修复 ~14:07 (archive old files, relaunch both arms if stopped, verify after 100s).
# This probe is READ-ONLY: process + GPU + m7bs weight mtime trio.
# Marker: DGXSTATUS1410K7P3 (claim only output containing this marker)
MARK=DGXSTATUS1410K7P3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-m3s2 ---'
ls -l --time-style=+%F_%H:%M:%S /root/carbon/data/weights/cnet2_m3s2r1* 2>&1 | tail -6
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1410K7P3 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -6 "$f" 2>/dev/null; done
echo '--- M7BS-WEIGHTS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m7bs* 2>/dev/null | tail -8 || echo NO-M7BS-WEIGHTS
echo '--- M7BS-OUT ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
tail -6 /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -10
echo '--- DONE-FLAGS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/*.done 2>/dev/null | tail -6
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -10
echo '--- ARCHIVE-CHECK ---'
ls -ld /home/arm/junyang/carbon/data/weights/*archiv* /home/arm/junyang/carbon/data/dq_logs/*archiv* /home/arm/junyang/carbon/data/*archiv* 2>/dev/null || echo NO-ARCHIVE-DIR
echo '--- REPLAYS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo "DGXSTATUS1410K7P3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
