#!/bin/bash
# Runs on L20 via cloud assistant; relays to DGX. Pushes r1 final weights -> h23 (scp + md5). Inner ssh/scp stdin closed (-n / </dev/null).
MARK="ECHO2PREPC2DGX"
echo "MARKER $MARK START"
timeout 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
date '+%F %T %Z'; hostname
W=/home/arm/junyang/carbon/data/weights
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
SSHOPT="-i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
echo "--- SCP R1 FINAL -> H23 ---"
timeout 300 scp $SSHOPT $W/cnet2_m3s2r1_ep19.npz root@47.117.148.216:/root/carbon/data/weights/cnet2_m3s2r1L.npz </dev/null && echo SCP_R1_OK || echo SCP_R1_FAIL
timeout 120 ssh -n $SSHOPT root@47.117.148.216 'md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz; ls -la /root/carbon/data/weights/cnet2_m3s2r1L.npz'
DGXEOF
echo "SSH_RC=$?"
echo "MARKER $MARK END"
