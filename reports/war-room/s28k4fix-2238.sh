#!/bin/bash
# s28 K4 weights top-up: copy the two K4v2 ep19 weights from the DGX workdir
# (they were never in s26stage) to spot-28, md5-verify both ends.
# Also identifies the +1 staging file (6039 vs 6038 at 19:50).
# Runs on L20 via cloud-assistant. Marker: S28K4FIX2238
MARK=S28K4FIX2238
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S28K4FIX2238
W=/home/arm/junyang/carbon/data/weights
ST=/home/arm/junyang/s26stage
TGT=47.116.17.244
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K4-SRC-CHECK ---'
ls -l "$W/cnet2_k4v2r1_fromstudio.npz" "$W/cnet2_k4v2g1_fromstudio.npz"
md5sum "$W/cnet2_k4v2r1_fromstudio.npz" "$W/cnet2_k4v2g1_fromstudio.npz"
md5sum "$W/cnet2_k4v2r1_fromstudio.npz" | grep -q "^344cfde939debb0de7f8834af069ea16 " && echo K4R1-SRC-OK || echo K4R1-SRC-MISMATCH
md5sum "$W/cnet2_k4v2g1_fromstudio.npz" | grep -q "^a594efd6cdb865d978e511c66d299206 " && echo K4G1-SRC-OK || echo K4G1-SRC-MISMATCH
echo '--- STAGE-PLUS-ONE ---'
find "$ST" -type f -newermt "2026-11-12 19:50:00" -printf "%T@ %p\n" 2>/dev/null | sort -n
echo '--- SCP-K4-TO-TARGET ---'
timeout -k 30 240 scp -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$W/cnet2_k4v2r1_fromstudio.npz" "$W/cnet2_k4v2g1_fromstudio.npz" root@"$TGT":/root/carbon/data/weights/
echo "SCP-RC=$?"
echo '--- K4-TGT-CHECK ---'
timeout -k 30 120 ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
C=/root/carbon
md5sum "$C/data/weights/cnet2_k4v2r1_fromstudio.npz" "$C/data/weights/cnet2_k4v2g1_fromstudio.npz"
md5sum "$C/data/weights/cnet2_k4v2r1_fromstudio.npz" | grep -q "^344cfde939debb0de7f8834af069ea16 " && echo K4R1-TGT-OK || echo K4R1-TGT-MISMATCH
md5sum "$C/data/weights/cnet2_k4v2g1_fromstudio.npz" | grep -q "^a594efd6cdb865d978e511c66d299206 " && echo K4G1-TGT-OK || echo K4G1-TGT-MISMATCH
echo -n "target-files="; find "$C" -type f | wc -l
echo -n "weights-files="; find "$C/data/weights" -type f | wc -l
TGTEOF
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
