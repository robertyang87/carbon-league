#!/bin/bash
# Locate cnet2_m7bsr1L.npz (or any m7bs weight) on DGX and on L20 (read-only listings).
# Marker: M7FIND2030G (claim only output containing this marker)
MARK=M7FIND2030G
echo "${MARK}_BEGIN $(date '+%F %T %Z')"
echo "--- L20 local weights (m7bs) ---"
ls -la /root/carbon/data/weights/ 2>/dev/null | grep -i m7bs || echo NONE_ON_L20_WEIGHTS
md5sum /root/carbon/data/weights/cnet2_m7bsr1L.npz 2>&1
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M7FIND2030G DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo "--- find *m7bs* under /home/arm ---"
timeout 120 find /home/arm -iname '*m7bs*' 2>/dev/null | head -60
echo "--- weights dirs m7 grep ---"
ls /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i m7 || echo NONE_CARBON_WEIGHTS
ls /home/arm/junyang/s26stage/data/weights/ 2>/dev/null | grep -i m7 || echo NONE_S26STAGE
ls /home/arm/junyang/carbon-repo/data/weights/ 2>/dev/null | grep -i m7 || echo NONE_REPO_WEIGHTS
echo "M7FIND2030G DGX-END"
DGXEOF
echo "${MARK}_END rc=$?"
