#!/bin/bash
# spot25 step F6: via DGX - read cnet2_search aux-load block on S25; check aux files on H24.
MARK="SPOT25F2K6"
echo "MARKER $MARK START"
date '+%F %T %Z'
timeout 180 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
H24=47.117.16.90
S25=47.116.17.244
echo "--- CNET2_SEARCH 470-500 ON S25 ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$S25" 'sed -n "470,500p" /root/carbon/code/cnet2_search.py'
echo "--- AUX FILES ON H24 ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'ls -la /root/carbon/data/weights/cnet2_s64.npz 2>/dev/null; md5sum /root/carbon/data/weights/cnet2_s64.npz 2>/dev/null || echo NO-S64-ON-H24'
DGXEOF
echo "DGX-SSH-RC=$?"
echo "MARKER $MARK END"
