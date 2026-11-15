#!/bin/bash
# Via L20 hop: DGX scp wr1/wg1 weights to h23, verify md5 on h23 side.
# Marker: DGXWXFER2016C (claim only output containing this marker)
MARK=DGXWXFER2016C
echo "${MARK}_BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXWXFER2016C DGX-BEGIN $(date '+%F %T %Z')"
W=/home/arm/junyang/carbon/data/weights
timeout -k 30 300 scp -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 $W/cnet2_m3s2wr1_fromstudio.npz $W/cnet2_m3s2wg1_fromstudio.npz root@47.117.148.216:/root/carbon/data/weights/
echo "scp rc=$?"
echo "--- verify on h23 ---"
timeout -k 10 60 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@47.117.148.216 'md5sum /root/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz /root/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz; ls -la /root/carbon/data/weights/ | grep fromstudio' 2>&1
echo "DGXWXFER2016C DGX-END"
DGXEOF
echo "${MARK}_END rc=$?"
