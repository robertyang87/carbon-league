#!/bin/bash
# M7V2 fix probe: tail DGX stage log + trainer status. Marker: __PROBE_MARK__
MARK=__PROBE_MARK__
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'tail -n 40 /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log; echo ---; pgrep -af "cnet2_train[.]py" || echo NO-TRAIN-PROC; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
echo "$MARK PROBE-END"
