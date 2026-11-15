#!/bin/bash
# L20-side launcher: push k4v2 DGX stage script to DGX, run detached.
MARK=K4V2LCH1
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '__STAGE_B64__' | base64 -d > /tmp/k4v2-dgx-stage.sh
wc -c /tmp/k4v2-dgx-stage.sh
timeout 60 scp -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new /tmp/k4v2-dgx-stage.sh arm@10.211.43.155:/tmp/k4v2-dgx-stage.sh && echo SCP_OK
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'mkdir -p /home/arm/junyang/carbon/data/dq_logs; nohup bash /tmp/k4v2-dgx-stage.sh > /home/arm/junyang/carbon/data/dq_logs/k4v2_stage_outer.log 2>&1 & sleep 6; pgrep -f "k4v2-dgx-stage[.]sh" | head -3; tail -5 /home/arm/junyang/carbon/data/dq_logs/k4v2_stage.log 2>/dev/null'
echo "$MARK SSH_RC=$? L20-END"
