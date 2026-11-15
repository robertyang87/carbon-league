#!/bin/bash
# DGX repo cut to main, trip 2 step 3: explicit-refspec fetch + checkout main. Marker: DGXMAIN2
MARK=DGXMAIN2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN2 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN2 NO-REPO"; exit 1; }
DST=/home/arm/junyang/carbon/data/attic/matches_m3s2_122_20261112.json
echo '--- PRE ---'
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
echo '--- PRE-STATUS (must be empty) ---'
git status --porcelain
if [ -n "$(git status --porcelain)" ]; then
  echo "DGXMAIN2 ABORT: worktree not clean at fix-trip start"
  exit 2
fi
DST_MD5=$(md5sum "$DST" | awk '{print $1}')
echo "attic_md5=$DST_MD5"
if [ "$DST_MD5" != "20d913c266bec53fcb88572e16eab2e1" ]; then
  echo "DGXMAIN2 ABORT: attic md5 mismatch"
  exit 2
fi
echo '--- REMOTE-CONFIG ---'
git config --get-all remote.origin.fetch
git branch -a
echo '--- FETCH-EXPLICIT ---'
timeout 60 git fetch origin +refs/heads/main:refs/remotes/origin/main
echo "fetch rc=$?"
git rev-parse origin/main
echo '--- CHECKOUT ---'
git checkout main
echo "checkout rc=$?"
echo '--- FF-MERGE ---'
git merge --ff-only origin/main
echo "ff-merge rc=$?"
echo '=== ATTEST ==='
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "head=$(git rev-parse HEAD)"
echo "status_lines=$(git status --porcelain | wc -l)"
git status --porcelain
git branch -vv
ls -l "$DST"
md5sum "$DST"
git log --oneline -3
echo "DGXMAIN2 FIX-DONE"
echo "DGXMAIN2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
