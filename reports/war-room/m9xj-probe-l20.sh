#!/bin/bash
# L20 wrapper: read-only DGX probe of the m9xj chain + training state.
# Marker passed per-run in the echoed line below (M9XJPRB...).
MARK=M9XJPRB_PLACEHOLDER
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M2=M9XJPRB_PLACEHOLDER
echo "$M2 DGX-BEGIN $(date '+%F %T %Z')"
D=/home/arm/junyang/carbon/data
echo "chain state: $(cat $D/m9xj_chain.state 2>/dev/null || echo NO-STATE)"
echo '--- chain log key lines ---'
grep -E 'COUNTS|INDEX-BUILT|SCAN-DROPPED|DS-M9XJ|WEIGHTED-REPORT|VAL-IDENTICAL|SMOKE|PUSH|AGGMD5|MD5-M|STAGE|FAIL|COUNTS-OK|dataset games|LAUNCH-OK|ATTESTATION|ep0 lines' $D/dq_logs/m9xj_chain.log $D/dq_logs/replica_chain_m9xj260.log 2>/dev/null | tail -40
echo '--- processes ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m9xj_chain' | grep -v grep || echo NO-PROC
echo '--- gpu ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- r1 log ---'
ls -l --time-style=+%F_%T $D/dq_logs/m9xj260r1.log 2>/dev/null; grep '"ep":' $D/dq_logs/m9xj260r1.log 2>/dev/null | tail -3
echo '--- g1 log ---'
ls -l --time-style=+%F_%T $D/dq_logs/m9xj260g1.log 2>/dev/null; grep '"ep":' $D/dq_logs/m9xj260g1.log 2>/dev/null | tail -3
echo '--- weights ---'
ls -lt --time-style=+%F_%T $D/weights/cnet2_m9xj260*.npz 2>/dev/null | head -8
echo '--- profile ---'
cat $D/m9xj_profile97.json 2>/dev/null
echo
echo "$M2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
