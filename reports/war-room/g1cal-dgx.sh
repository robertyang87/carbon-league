#!/bin/bash
# g1cal DGX probe (runs on L20, hops to DGX): md5 of m3s2g1 weights on DGX
# + repo HEAD, for comparison against h23 pre-staged weights.
MARK="G1CALDGX7"
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- DGX WEIGHTS m3s2g1 ---'
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "G1CALDGX7 DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo '--- weights ---'
ls -la /home/arm/junyang/carbon/data/weights/ 2>&1 | grep -i "m3s2g1" || echo NO-M3S2G1-WEIGHTS-DGX
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2g1* 2>&1
echo '--- repo HEAD ---'
git -C /home/arm/junyang/carbon-repo rev-parse HEAD 2>&1 | head -2
git -C /home/arm/junyang/carbon-repo log --oneline -3 2>&1 | head -5
echo '--- repo eval_v53 md5 ---'
md5sum /home/arm/junyang/carbon-repo/code/eval_v53.py 2>&1
echo "G1CALDGX7 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
