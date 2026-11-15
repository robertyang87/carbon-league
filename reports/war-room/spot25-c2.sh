#!/bin/bash
# spot25 step C2: L20 -> DGX -> h24 code breakdown + md5s of weights/wrappers.
MARK="SPOT25C2K5"
echo "MARKER $MARK START"
date '+%F %T %Z'
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
H24=47.117.16.90
echo "--- H24 CODE BREAKDOWN ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'du -sh /root/carbon/code/* 2>/dev/null | sort -rh | head -15'
echo "--- H24 MD5 ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'md5sum /root/carbon/data/weights/cnet2_m3w5L.npz /root/carbon/data/weights/cnet2_m3w3L.npz /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/code/exp/echo2.py /root/carbon/code/exp/echo_e0.py /root/carbon/code/exp/m3s2r1.py'
DGXEOF
echo "DGX-SSH-RC=$?"
echo "MARKER $MARK END"
