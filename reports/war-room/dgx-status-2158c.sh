#!/bin/bash
# DGX micro-probe 2158c: g1 final val (train log tail), r1 latest epoch, chain outer tail.
# Read-only. Marker: DGXSTATUS2158V
MARK=DGXSTATUS2158V
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2158V DGX-BEGIN $(date '+%F %T %Z')"
echo '--- G1-TRAIN-LOG-TAIL ---'
tail -25 /home/arm/junyang/carbon/data/dq_logs/m9xj320g1.log 2>/dev/null || echo NO-G1-LOG
echo '--- R1-TRAIN-LOG-TAIL ---'
tail -12 /home/arm/junyang/carbon/data/dq_logs/m9xj320r1.log 2>/dev/null || echo NO-R1-LOG
echo '--- CHAIN-OUTER-TAIL ---'
tail -20 /home/arm/junyang/carbon/data/dq_logs/m320n_chain_outer.log 2>/dev/null || echo NO-OUTER-LOG
echo '--- WEIGHTS-NEWEST ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m9xj320*.npz | head -6
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep cnet2_train | grep -v grep || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "DGXSTATUS2158V DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
