#!/bin/bash
# DGX status probe 11 (scheduled 15:13 run): probe self-stop watch + Carago landing watch + distill r1 return watch.
# Last direct read 15:00:21-27: M7Bs weights all on L20 + md5 closed; distill g1 md5 6c3885c6 closed.
# War room 15:04 correction: the 14:59 "k4r1 probe" was actually the CPU generation-rate probe
# (probe_n1r, 32 workers, timeout 330, self-stop ~15:04:30); K4 probe still waits for h24 calibration.
# War room killed 2 stale ZT procs (ship_m7bs.sh 99500, scp 103651) at 15:04.
# Open items: Carago corpus landing (war-room est ~15:02, not landed at 15:00:27);
# Studio distill r1 est ~15:07 done then auto-return to DGX; Fable replay pull run 43172461506 in flight.
# This probe is READ-ONLY: process + GPU + weight/log mtime trio, md5 reads only.
# Marker: DGXSTATUS1513K7Q2 (claim only output containing this marker)
MARK=DGXSTATUS1513K7Q2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-WEIGHTS-MD5 ---'
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1513K7Q2 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|train|carago|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- PROBE-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'probe_n1r|bigeval' | grep -v grep || echo NO-PROBE-PROC
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log; do echo "== $f"; ls -l --time-style=+%F_%H:%M:%S "$f" 2>/dev/null; tail -3 "$f" 2>/dev/null; done
echo '--- WEIGHTS-NEWEST-ALL ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -12
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -8
echo '--- DATA-DIR-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -15
echo '--- CARAGO-SEARCH ---'
find /home/arm/junyang/carbon/data -maxdepth 2 -iname '*carago*' 2>/dev/null | head -10; echo "carago-find-done"
find /home/arm/junyang/carbon -maxdepth 3 -iname '*fable*' 2>/dev/null | head -10; echo "fable-find-done"
echo '--- REPLAYS ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays_dir_count='
ls /home/arm/junyang/carbon/data/m3s2_corpus/replays/ 2>/dev/null | wc -l
echo "DGXSTATUS1513K7Q2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
