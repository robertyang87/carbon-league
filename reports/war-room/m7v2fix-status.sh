#!/bin/bash
# M7V2 status probe: DGX trainer PIDs + GPU + trainer log tails + stage tail.
# Marker: __PROBE_MARK__
MARK=__PROBE_MARK__
echo "$MARK STATUS-BEGIN $(date '+%F %T %Z')"
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'echo DGX-HOST $(hostname) $(date "+%F %T %Z"); echo --- train-procs ---; pgrep -af "cnet2_train[.]py" || echo NO-TRAIN-PROC; echo --- stage-proc ---; pgrep -af "m7v2fix-dgx" || echo NO-STAGE-PROC; echo --- gpu ---; nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader; echo --- r1-log-tail ---; tail -n 2 /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log 2>&1 | cut -c1-300; echo --- g1-log-tail ---; tail -n 2 /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log 2>&1 | cut -c1-300; echo --- stage-log-tail ---; tail -n 12 /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log | cut -c1-250'
echo "$MARK STATUS-END"
