#!/bin/bash
# DGX HEAD mini-probe 2128 (follow-up: main probe log truncated before HEAD section).
# Read-only. Marker: DGXSTATUS2128HEAD
MARK=DGXSTATUS2128HEAD
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2128HEAD DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD && git log -2 --format='%h %ci %s' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1 | head -2 && timeout 60 git rev-list --count HEAD..origin/cnet2-search-v12 2>&1
echo "DGXSTATUS2128HEAD DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
