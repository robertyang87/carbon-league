#!/bin/bash
# War-room M7Bs v2 FIX launcher: push fix DGX script to DGX, run detached.
# The DGX script body arrives base64-embedded; the driver substitutes the
# token in the echo line below (it must occur exactly once in this file).
# Marker: M7V2FX7Q2
MARK=M7V2FX7Q2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '__DGX_B64__' | base64 -d > /tmp/m7v2fix-dgx.sh
wc -c /tmp/m7v2fix-dgx.sh
timeout 60 scp -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new /tmp/m7v2fix-dgx.sh arm@10.211.43.155:/tmp/m7v2fix-dgx.sh && echo SCP_OK
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'nohup bash /tmp/m7v2fix-dgx.sh > /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log 2>&1 & sleep 8; pgrep -f "m7v2fix-dgx[.]sh" | head -3; tail -5 /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log'
echo "$MARK SSH_RC=$? L20-END"
