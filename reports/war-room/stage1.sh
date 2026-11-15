#!/bin/bash
# War-room staging stage1 launcher: push stage1-dgx.sh to DGX, run it detached.
# The DGX script body arrives base64-embedded (driver substitutes __DGX_B64__).
# Marker: STAGE1CFZ3
MARK=STAGE1CFZ3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '__DGX_B64__' | base64 -d > /tmp/stage1-dgx.sh
wc -c /tmp/stage1-dgx.sh
timeout 60 scp -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new /tmp/stage1-dgx.sh arm@10.211.43.155:/tmp/stage1-dgx.sh && echo SCP_OK
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'nohup bash /tmp/stage1-dgx.sh > /home/arm/junyang/carbon/data/dq_logs/stage1.log 2>&1 & sleep 8; pgrep -f "stage1-dgx[.]sh" | head -3; tail -5 /home/arm/junyang/carbon/data/dq_logs/stage1.log'
echo "$MARK SSH_RC=$? L20-END"
