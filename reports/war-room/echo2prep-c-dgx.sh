#!/bin/bash
# Runs on L20 via cloud assistant; relays to DGX. Pushes: W5L weights -> h24, r1 final -> h23 (scp + md5).
MARK="ECHO2PREPQ9XDGX"
echo "MARKER $MARK START"
timeout 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
date '+%F %T %Z'; hostname
W=/home/arm/junyang/carbon/data/weights
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
SSHOPT="-i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
echo "--- DGX W5 MD5S ---"
timeout 120 md5sum $W/cnet2_m3w5_ep19.npz $W/cnet2_m3w5_last.npz $W/cnet2_m3w5.npz
SRC=""
for f in cnet2_m3w5_ep19.npz cnet2_m3w5_last.npz; do
  h=$(md5sum $W/$f | cut -d' ' -f1)
  if [ "$h" = "82fde46b2633950d3cbb29e8cacbfdc2" ]; then SRC=$f; fi
done
echo "W5_SRC=$SRC"
if [ -n "$SRC" ]; then
  echo "--- SCP W5L -> H24 ---"
  timeout 300 scp $SSHOPT $W/$SRC root@47.117.16.90:/root/carbon/data/weights/cnet2_m3w5L.npz && echo SCP_W5_OK || echo SCP_W5_FAIL
  timeout 120 ssh $SSHOPT root@47.117.16.90 'md5sum /root/carbon/data/weights/cnet2_m3w5L.npz; ls -la /root/carbon/data/weights/cnet2_m3w5L.npz'
else
  echo "W5_SRC_NOT_FOUND_SKIP_PUSH"
fi
echo "--- SCP R1 FINAL -> H23 ---"
timeout 300 scp $SSHOPT $W/cnet2_m3s2r1_ep19.npz root@47.117.148.216:/root/carbon/data/weights/cnet2_m3s2r1L.npz && echo SCP_R1_OK || echo SCP_R1_FAIL
timeout 120 ssh $SSHOPT root@47.117.148.216 'md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz; ls -la /root/carbon/data/weights/cnet2_m3s2r1L.npz'
DGXEOF
echo "SSH_RC=$?"
echo "MARKER $MARK END"
