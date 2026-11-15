#!/bin/bash
# L20-side wrapper: ssh to DGX and run the pull script there.
MARK="DGXPULL7K2Q9"
echo "MARKER $MARK START"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK="DGXPULL7K2Q9"
echo "MARKER $MARK DGX-START"
cd /home/arm/junyang/carbon-repo || { echo "MARKER $MARK CD-FAIL"; exit 1; }
echo "--- BEFORE HEAD ---"
git rev-parse HEAD
echo "--- LS-REMOTE ---"
timeout 60 git ls-remote origin cnet2-search-v12
echo "--- PULL ---"
timeout 120 git pull --ff-only origin cnet2-search-v12
echo "pull rc=$?"
echo "--- AFTER HEAD ---"
git rev-parse HEAD
echo "--- LOG -6 ---"
git log --oneline -6
echo "--- STATUS ---"
git status --short
echo "--- END STATUS (empty means clean) ---"
echo "--- GIT LEFTOVER PROCS ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo "no git fetch/pull/checkout procs"
echo "MARKER $MARK DGX-END"
DGXEOF
echo "MARKER $MARK L20-END rc=$?"
