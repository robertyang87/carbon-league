#!/bin/bash
# Studio md5 probe 1512b: Studio-side md5 of r1 final weights for DGX-side cross-check (g1 precedent)
# Marker: STUDIOSTAT1512MD5A (claim only output containing this marker)
MARK=STUDIOSTAT1512MD5A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1512MD5A STUDIO-BEGIN $(date '+%F %T %Z')"
cd /Users/arm/junyang/carbon/data/weights
md5 -q cnet2_distill_r1_ep19.npz
md5 -q cnet2_distill_r1_last.npz
ls -l cnet2_distill_r1_ep19.npz cnet2_distill_r1_last.npz
echo "STUDIOSTAT1512MD5A STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
