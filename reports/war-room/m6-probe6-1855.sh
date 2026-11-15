#!/bin/bash
# M6 probe 6: read the M7V2 fix stage script head (sync + index-rebuild incl.
# teacher-id resolution) so the M6 chain replicates it exactly.
# Marker: M6PRB61855K3
MARK=M6PRB61855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB61855K3 DGX-BEGIN $(date '+%F %T %Z')"
sed -n '1,130p' /tmp/m7v2fix-dgx.sh 2>&1
echo "M6PRB61855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
