#!/bin/bash
# DGX status probe 12 (scheduled 15:28 run): Carago landing watch + distill r1 md5 reconciliation + r1fable probe watch.
# Last direct read 15:16:26-31: GPU 0%/37C, no training; distill r1 landed 15:06 (13,968,960 B), md5 NOT checked.
# War room update (15:26): Carago pull run 43173011168 completed success 15:22:24, data/carago_corpus 183 replays landed in repo.
# War room next: wait r1fable probe (dispatched 15:22:44, in flight) then dispatch DGX build index+dataset+train both arms.
# User asked war room (15:28): whether ladder-accumulated corpus of in-library models should be added and substitutes trained - war room's call, not ours.
# This probe is READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1528M4P9 (claim only output containing this marker)
MARK=DGXSTATUS1528M4P9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- DGX-DISTILL-R1-MD5 (via ssh, expect 93061c69d6bc7f9f3d0468f60ce54974 per Studio original) ---'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1528M4P9 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- PROBE-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'probe_n1r|bigeval' | grep -v grep || echo NO-PROBE-PROC
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -2 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
md5sum /home/arm/junyang/carbon/data/weights/cnet2_distill_r1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_distill_g1_fromstudio.npz 2>&1
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -6
echo '--- DATA-DIR-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -15
echo '--- CARAGO-SEARCH ---'
find /home/arm/junyang/carbon/data -maxdepth 2 -iname '*carago*' 2>/dev/null | head -10; echo "carago-find-done"
find /home/arm/junyang/carbon -maxdepth 3 -iname '*fable*' 2>/dev/null | head -10; echo "fable-find-done"
echo '--- REPLAYS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo "DGXSTATUS1528M4P9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
