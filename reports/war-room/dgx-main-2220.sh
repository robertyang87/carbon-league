#!/bin/bash
# DGX repo cut to main (owner rule 21:51). Read-only except fetch+checkout; skips checkout if worktree dirty.
# Marker: DGXMAIN2220
MARK=DGXMAIN2220
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN2220 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN2220 NO-REPO"; exit 1; }
echo '--- PRE-STATE ---'
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git log --oneline -3
echo '--- STATUS-PORCELAIN ---'
git status --porcelain
if [ -n "$(git status --porcelain)" ]; then
  echo "DGXMAIN2220 DIRTY-SKIP: worktree not clean, checkout NOT attempted"
else
  echo '--- FETCH ---'
  timeout 60 git fetch origin main
  echo "fetch rc=$?"
  echo '--- REMOTE-TIP ---'
  git ls-remote origin main
  echo '--- CHECKOUT ---'
  git checkout main
  echo "checkout rc=$?"
  git merge --ff-only origin/main
  echo "ff-merge rc=$?"
  echo '--- POST-STATE ---'
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git log --oneline -3
  echo '--- STATUS-AFTER ---'
  git status --short
  echo "DGXMAIN2220 DONE"
fi
echo "DGXMAIN2220 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
