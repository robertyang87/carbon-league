#!/bin/bash
# H23 prestage step C: DGX -> h23 scp of m3s2g1 weights via relay key + verify on h23.
MARK=PREH23C7K9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PREH23C7K9 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- SCP DGX->H23 ---'
timeout 300 scp -i /home/arm/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new /home/arm/junyang/carbon/data/weights/cnet2_m3s2g1_ep19.npz root@47.117.148.216:/root/carbon/data/weights/cnet2_m3s2g1L.npz 2>&1
echo "scp rc=$?"
echo '--- VERIFY ON H23 ---'
timeout 60 ssh -i /home/arm/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new root@47.117.148.216 'ls -l /root/carbon/data/weights/cnet2_m3s2g1L.npz; md5sum /root/carbon/data/weights/cnet2_m3s2g1L.npz' 2>&1
echo '--- SRC MD5 AGAIN ---'
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2g1_ep19.npz
echo "PREH23C7K9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
