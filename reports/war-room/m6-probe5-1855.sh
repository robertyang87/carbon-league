#!/bin/bash
# M6 probe 5: print the K4v2 DGX stage script (contains the proven index-build
# step) and how the M7V2 fix script resolves the teacher strategy id.
# Marker: M6PRB51855K3
MARK=M6PRB51855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB51855K3 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K4V2-STAGE-SCRIPT ---'
cat /tmp/k4v2-dgx-stage.sh 2>&1
echo '--- M7V2FIX-TEACHER-ID ---'
grep -n "teacher" /tmp/m7v2fix-dgx.sh 2>&1 | head -25
echo "M6PRB51855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
