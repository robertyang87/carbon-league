#!/bin/bash
# H23 prestage probe A: DGX source weights + relay-key reachability to h23. READ-ONLY.
MARK=PREH23A7K9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PREH23A7K9 DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo '--- SRC-WEIGHT ---'
ls -l /home/arm/junyang/carbon/data/weights/cnet2_m3s2g1_ep19.npz 2>&1
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2g1_ep19.npz 2>&1
echo '--- RELAY-KEY ---'
ls -l /home/arm/.ssh/id_ed25519_carbon_relay 2>&1
echo '--- H23-REACH (relay key, read-only hostname) ---'
timeout 30 ssh -i /home/arm/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new root@47.117.148.216 'hostname; echo H23-REACH-OK' 2>&1
echo "PREH23A7K9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
