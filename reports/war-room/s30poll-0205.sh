#!/bin/bash
# s30 poll: read the DGX-side transfer worker log. Marker: S30POLL0205
echo "S30POLL0205 L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'cat /home/arm/junyang/s30xfer.log 2>/dev/null || echo NO-LOG-YET'
echo "S30POLL0205 L20-END rc=$?"
