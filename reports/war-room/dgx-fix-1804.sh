#!/bin/bash
# Read-only urgent check (owner order 18:04): is m7bsv2 training live yet? Trio + executor logs + GPU.
# Marker: DGXFIX1804Z9K2
MARK=DGXFIX1804Z9K2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXFIX1804Z9K2 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bsv2|m7v2' | grep -v grep || echo NO-TRAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -8 || echo NO-PYTHON
echo '--- M7BSV2-LOGS ---'
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/ | head -12
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; tail -3 "$f" 2>/dev/null || echo NO-LOG; done
echo '--- M7V2-EXECUTOR-LOGS ---'
for f in $(ls -t /home/arm/junyang/carbon/data/dq_logs/m7v2*.log /home/arm/junyang/carbon/data/m7v2*.log 2>/dev/null | head -3); do echo "== $f"; tail -6 "$f"; done
echo '--- M7BSV2-WEIGHTS ---'
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/ | grep -i 'm7bsv2' | head -6; echo weights-grep-done
echo '--- DS-M7BSV2 ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/ds_m7bs_v2/meta.json 2>/dev/null || echo NO-DS-META
echo "DGXFIX1804Z9K2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
