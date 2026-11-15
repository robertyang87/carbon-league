#!/bin/bash
# DGX status probe 5 (confirm): M7Bs really not running? 2026-11-12 ~14:05 CST
# Marker: DGXSTATUS1405MX2K (claim only output containing this marker)
MARK=DGXSTATUS1405MX2K
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1405MX2K DGX-BEGIN $(date '+%F %T %Z')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- ALL-PYTHON-PROC ---'
ps -eo pid,etime,args | grep -E 'python|train' | grep -v grep || echo NO-PYTHON-PROC
echo '--- DQ-LOGS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -8
echo '--- M7BS-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsr1.log /home/arm/junyang/carbon/data/dq_logs/m7bsg1.log /home/arm/junyang/carbon/data/dq_logs/build_m7bs.log; do echo "== $f"; tail -4 "$f" 2>/dev/null; done
echo '--- M7BS-WEIGHTS-NEWEST ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
echo '--- HANDOFF-SCRIPTS ---'
ls /home/arm/junyang/carbon/code/ 2>/dev/null | grep -i -E 'm7bs|handoff|watch|ship|auto' || echo NO-HANDOFF-SCRIPT-MATCH
echo "DGXSTATUS1405MX2K DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
