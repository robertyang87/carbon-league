#!/bin/bash
# DGX status probe 13 (scheduled 15:43 run): war-room recon probe #2 watch + Carago/Fable landing watch.
# War room latest (read 15:43): bounded recon probe #2 dispatched 15:35, expected ~15:42 return;
# it also moves Carago (183) + Fable (182) corpora into DGX data dir. War room then dispatches
# DGX Carago index+dataset+both-arms train, then Fable dataset -> Studio.
# Last direct read 15:30:04: GPU 0%/37C, no training; distill r1 md5 closed; Carago repo-only.
# This probe is READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1543K7Q2 (claim only output containing this marker)
MARK=DGXSTATUS1543K7Q2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo '--- DGX (via ssh) ---'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1543K7Q2 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- PROBE-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'probe_n1r|bigeval|r1fable' | grep -v grep || echo NO-PROBE-PROC
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -2 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
md5sum /home/arm/junyang/carbon/data/weights/cnet2_distill_r1_fromstudio.npz /home/arm/junyang/carbon/data/weights/cnet2_distill_g1_fromstudio.npz 2>&1
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -6
echo '--- DATA-DIR-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -20
echo '--- CARAGO-FABLE-SEARCH ---'
find /home/arm/junyang/carbon/data -maxdepth 2 -iname '*carago*' 2>/dev/null | head -10; echo "carago-find-done"
find /home/arm/junyang/carbon -maxdepth 3 -iname '*fable*' 2>/dev/null | head -10; echo "fable-find-done"
for d in /home/arm/junyang/carbon/data/carago_corpus /home/arm/junyang/carbon/data/fable_corpus; do
  echo "== $d"; ls "$d" 2>/dev/null | head -3; echo -n 'count='; find "$d" -type f 2>/dev/null | wc -l
done
echo '--- REPLAYS-M3S2 ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo '--- RECENTLY-MODIFIED-UNDER-DATA (30min) ---'
find /home/arm/junyang/carbon/data -mmin -30 -type f 2>/dev/null | head -20; echo "recent-find-done"
echo "DGXSTATUS1543K7Q2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
