#!/bin/bash
# War-room M7Bs v2 launcher: push m7v2-dgx.sh to DGX, run it detached.
# The DGX script body arrives base64-embedded (driver substitutes the token below).
# Marker: M7V2K8R3
MARK=M7V2K8R3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '__DGX_B64__' | base64 -d > /tmp/m7v2-dgx.sh
wc -c /tmp/m7v2-dgx.sh
timeout 60 scp -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new /tmp/m7v2-dgx.sh arm@10.211.43.155:/tmp/m7v2-dgx.sh && echo SCP_OK
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'nohup bash /tmp/m7v2-dgx.sh > /home/arm/junyang/carbon/data/dq_logs/m7v2.log 2>&1 & sleep 8; pgrep -f "m7v2-dgx[.]sh" | head -3; tail -5 /home/arm/junyang/carbon/data/dq_logs/m7v2.log'
echo "$MARK SSH_RC=$? L20-END"
