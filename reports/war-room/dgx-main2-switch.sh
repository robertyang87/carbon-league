#!/bin/bash
# DGX repo cut to main, trip 2 step 4: create local main at origin/main, switch, restore old branch pointer. Marker: DGXMAIN2
MARK=DGXMAIN2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN2 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN2 NO-REPO"; exit 1; }
DST=/home/arm/junyang/carbon/data/attic/matches_m3s2_122_20261112.json
echo '--- PRE ---'
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --porcelain
if [ -n "$(git status --porcelain)" ]; then
  echo "DGXMAIN2 ABORT: worktree not clean at switch-trip start"
  exit 2
fi
DST_MD5=$(md5sum "$DST" | awk '{print $1}')
echo "attic_md5=$DST_MD5"
if [ "$DST_MD5" != "20d913c266bec53fcb88572e16eab2e1" ]; then
  echo "DGXMAIN2 ABORT: attic md5 mismatch"
  exit 2
fi
echo '--- REFRESH-ORIGIN-MAIN ---'
timeout 60 git fetch origin +refs/heads/main:refs/remotes/origin/main
echo "fetch rc=$?"
git rev-parse origin/main
timeout 60 git ls-remote origin main
echo '--- CREATE-AND-SWITCH ---'
git checkout -b main origin/main
echo "checkout-b rc=$?"
echo "branch_now=$(git rev-parse --abbrev-ref HEAD)"
echo '--- FF-CHECK ---'
git merge --ff-only origin/main
echo "ff-merge rc=$?"
echo '--- RESTORE-OLD-BRANCH-POINTER ---'
git rev-parse origin/cnet2-search-v12
git branch -f cnet2-search-v12 origin/cnet2-search-v12
echo "branch-f rc=$?"
git rev-parse cnet2-search-v12
echo '=== ATTEST ==='
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "head=$(git rev-parse HEAD)"
echo "status_lines=$(git status --porcelain | wc -l)"
git status --porcelain
git branch -vv
ls -l "$DST"
md5sum "$DST"
git log --oneline -3
echo "DGXMAIN2 SWITCH-DONE"
echo "DGXMAIN2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
