#!/bin/bash
# DGX status probe 7 (scheduled 14:13 run): M7Bs progress check after confirmed restart.
# War-room relaunched both arms ~14:07:41; last direct read 14:12:31 (r1 ep2, g1 ep4, GPU 96%).
# This probe is READ-ONLY: process + GPU + m7bs weight/log mtime trio.
# Marker: DGXSTATUS1417M3K8 (claim only output containing this marker)
MARK=DGXSTATUS1417M3K8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-m3s2r1 ---'
ls -l --time-style=+%F_%H:%M:%S /root/carbon/data/weights/cnet2_m3s2r1* 2>&1 | tail -6
echo '--- L20-WEIGHTS-m3s2g1 ---'
ls -l --time-style=+%F_%H:%M:%S /root/carbon/data/weights/cnet2_m3s2g1* 2>&1 | tail -4
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1417M3K8 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -8 "$f" 2>/dev/null; done
echo '--- M7BS-WEIGHTS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m7bs* 2>/dev/null | tail -10 || echo NO-M7BS-WEIGHTS
echo '--- M7BS-OUT ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/m7bs.out 2>/dev/null
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -10
echo '--- DONE-FLAGS ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/*.done 2>/dev/null | tail -6
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -10
echo '--- REPLAYS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo "DGXSTATUS1417M3K8 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
