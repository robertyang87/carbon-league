#!/bin/bash
MARK=K4V2PRB1
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'cat /home/arm/junyang/carbon/data/dq_logs/k4v2_stage.log; echo ---STAGE-PROC---; pgrep -af "k4v2-dgx-stage[.]sh" || echo STAGE-NOT-RUNNING; echo ---GPU---; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
echo "$MARK PROBE-END"
