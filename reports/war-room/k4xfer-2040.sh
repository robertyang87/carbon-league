#!/bin/bash
# Relay k4v2r1/k4v2g1 weights DGX -> h23, md5-verify both ends; also check E0 weight on h23.
# Marker: K4XFER2040K (claim only output containing this marker)
MARK=K4XFER2040K
echo "${MARK}_BEGIN $(date '+%F %T %Z')"
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "K4XFER2040K DGX-BEGIN $(date '+%F %T %Z')"
W=/home/arm/junyang/carbon/data/weights
for f in cnet2_k4v2r1_fromstudio.npz cnet2_k4v2g1_fromstudio.npz; do
  if [ -f "$W/$f" ]; then md5sum "$W/$f"; else echo "MISSING_ON_DGX $f"; fi
done
for f in cnet2_k4v2r1_fromstudio.npz cnet2_k4v2g1_fromstudio.npz; do
  timeout -k 30 300 scp -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 "$W/$f" root@47.117.148.216:/root/carbon/data/weights/
  echo "relay $f rc=$?"
done
echo "--- verify on h23 (k4 weights + E0 weight) ---"
timeout -k 10 90 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@47.117.148.216 'md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz /root/carbon/data/weights/cnet2_m3w3L.npz; ls -la /root/carbon/data/weights/cnet2_k4v2*.npz' 2>&1
echo "K4XFER2040K DGX-END"
DGXEOF
echo "${MARK}_END rc=$?"
