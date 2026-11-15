#!/bin/bash
# Via L20 hop: locate wr1/wg1 weights on DGX + md5; test DGX relay-key ssh to h23.
# Marker: DGXWRELAY2013B (claim only output containing this marker)
MARK=DGXWRELAY2013B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXWRELAY2013B DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo "--- find weights ---"
find /home/arm/junyang -name 'cnet2_m3s2wr1_fromstudio.npz' -o -name 'cnet2_m3s2wg1_fromstudio.npz' 2>/dev/null
echo "--- md5 ---"
for f in $(find /home/arm/junyang -name 'cnet2_m3s2wr1_fromstudio.npz' -o -name 'cnet2_m3s2wg1_fromstudio.npz' 2>/dev/null); do md5sum "$f"; ls -la "$f"; done
echo "--- relay key ---"
ls -la ~/.ssh/ | grep -i relay
echo "--- test ssh DGX->h23 (47.117.148.216) ---"
timeout -k 10 60 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new root@47.117.148.216 'hostname; date "+%F %T %Z"' 2>&1
echo "DGXWRELAY2013B DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
