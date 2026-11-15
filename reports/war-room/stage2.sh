#!/bin/bash
# War-room staging stage2 launcher: push stage2-dgx.sh to DGX, run it detached.
# The DGX script body arrives base64-embedded (driver substitutes __DGX_B64__).
# Marker: STAGE2CFZ4
MARK=STAGE2CFZ4
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '__DGX_B64__' | base64 -d > /tmp/stage2-dgx.sh
wc -c /tmp/stage2-dgx.sh
timeout 60 scp -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new /tmp/stage2-dgx.sh arm@10.211.43.155:/tmp/stage2-dgx.sh && echo SCP_OK
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'nohup bash /tmp/stage2-dgx.sh > /home/arm/junyang/carbon/data/dq_logs/stage2.log 2>&1 & sleep 8; pgrep -f "stage2-dgx[.]sh" | head -3; tail -5 /home/arm/junyang/carbon/data/dq_logs/stage2.log'
echo "$MARK SSH_RC=$? L20-END"
