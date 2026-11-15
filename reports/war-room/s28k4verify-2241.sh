#!/bin/bash
# s28 K4 target re-verify (retry of the silent check in s28k4fix-2238:
# that run used `ssh -n` with a heredoc-fed remote script, so the remote
# bash got EOF and printed nothing -- known lesson, -n removed here).
# Runs on L20 via cloud-assistant. Marker: S28K4VRF2241
MARK=S28K4VRF2241
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S28K4VRF2241
TGT=47.116.17.244
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
C=/root/carbon
ls -l "$C/data/weights/cnet2_k4v2r1_fromstudio.npz" "$C/data/weights/cnet2_k4v2g1_fromstudio.npz"
md5sum "$C/data/weights/cnet2_k4v2r1_fromstudio.npz" "$C/data/weights/cnet2_k4v2g1_fromstudio.npz"
md5sum "$C/data/weights/cnet2_k4v2r1_fromstudio.npz" | grep -q "^344cfde939debb0de7f8834af069ea16 " && echo K4R1-TGT-OK || echo K4R1-TGT-MISMATCH
md5sum "$C/data/weights/cnet2_k4v2g1_fromstudio.npz" | grep -q "^a594efd6cdb865d978e511c66d299206 " && echo K4G1-TGT-OK || echo K4G1-TGT-MISMATCH
echo -n "target-files="; find "$C" -type f | wc -l
echo -n "weights-files="; find "$C/data/weights" -type f | wc -l
TGTEOF
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
