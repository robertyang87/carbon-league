#!/bin/bash
# DGX status probe 10 (scheduled 14:58 run): DGX idle watch + L20 weight landing close-out.
# Last direct read 14:45:11: M7Bs BOTH DONE (g1 14:27 best 0.8615; r1 ~14:35 best 0.8629);
# DGX idle (GPU 0%/37C); weights re-sent via public SSH 14:51, md5 closed (m3s2r1L b6db08c9,
# m7bsr1L 0835830f, m7bsg1L 2c8ca0ae) per war room; distill_g1_fromstudio.npz landed DGX 14:42.
# Open items: m7bsg1L rsync final acceptance on L20; distill_g1 DGX-side md5; Carago re-pull ~15:02.
# This probe is READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1458QZ9P (claim only output containing this marker)
MARK=DGXSTATUS1458QZ9P
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS ---'
ls -l --time-style=+%F_%H:%M:%S /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- L20-RSYNC-SCP ---'
ps -eo pid,etime,args | grep -E 'rsync|scp' | grep -v grep || echo NO-RSYNC-SCP
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1458QZ9P DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- SCP-SHIP ---'
ps -eo pid,etime,args | grep -E 'scp|rsync|ship' | grep -v grep || echo NO-SCP-SHIP
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -5 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -12
echo '--- DISTILL-G1-MD5 ---'
md5sum /home/arm/junyang/carbon/data/weights/cnet2_distill_g1_fromstudio.npz 2>&1
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -8
echo '--- DATA-DIR-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -12
echo '--- REPLAYS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo "DGXSTATUS1458QZ9P DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
