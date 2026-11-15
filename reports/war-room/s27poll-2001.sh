#!/bin/bash
# s27 poll v2: read the DGX-side transfer worker log. Marker: S27POLL2001K
echo "S27POLL2001K L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'cat /home/arm/junyang/s27xfer.log 2>/dev/null || echo NO-LOG-YET'
echo "S27POLL2001K L20-END rc=$?"
