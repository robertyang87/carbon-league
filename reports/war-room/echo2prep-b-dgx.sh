#!/bin/bash
# Runs on L20 via cloud assistant; relays to DGX. READ ONLY: r1 ep19 md5 on DGX (source for h23 pre-stage).
MARK="ECHO2PREPM3WDGX"
echo "MARKER $MARK START"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
date '+%F %T %Z'; hostname
echo "--- R1 EP19 ON DGX ---"
ls -la /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_ep19.npz /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1L.npz 2>/dev/null
timeout 120 md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_ep19.npz 2>/dev/null
echo "--- M3W5 ON DGX ---"
ls -la /home/arm/junyang/carbon/data/weights/ | grep -i m3w5 || echo NO_M3W5_ON_DGX
DGXEOF
echo "SSH_RC=$?"
echo "MARKER $MARK END"
