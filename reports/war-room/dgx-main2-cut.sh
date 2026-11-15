#!/bin/bash
# DGX repo cut to main, trip 2 step 2: move stray file to attic, then fetch+checkout+ff-merge. Marker: DGXMAIN2
MARK=DGXMAIN2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN2 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN2 NO-REPO"; exit 1; }
SRC=code/matches_d679ed45ab595e8b7bb1ea51c3451154.json
DST=/home/arm/junyang/carbon/data/attic/matches_m3s2_122_20261112.json
echo '--- PRE-STATE ---'
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
echo '--- PRE-STATUS ---'
git status --porcelain
# Safety gate: exactly one dirty line and it must be the known file, md5 must match
NLINES=$(git status --porcelain | wc -l)
ONLY=$(git status --porcelain)
SRC_MD5=$(md5sum "$SRC" | awk '{print $1}')
echo "dirty_lines=$NLINES src_md5=$SRC_MD5"
if [ "$NLINES" != "1" ] || [ "$ONLY" != "?? $SRC" ] || [ "$SRC_MD5" != "20d913c266bec53fcb88572e16eab2e1" ]; then
  echo "DGXMAIN2 ABORT: pre-check mismatch, nothing touched"
  exit 2
fi
echo '--- MOVE-TO-ATTIC ---'
mkdir -p /home/arm/junyang/carbon/data/attic
mv "$SRC" "$DST"
echo "mv rc=$?"
ls -l "$DST"
md5sum "$DST"
echo '--- STATUS-AFTER-MOVE ---'
git status --porcelain
echo "(empty above = clean)"
echo '--- FETCH ---'
timeout 60 git fetch origin main
echo "fetch rc=$?"
echo '--- CHECKOUT ---'
git checkout main
echo "checkout rc=$?"
git merge --ff-only origin/main
echo "ff-merge rc=$?"
echo '=== ATTEST ==='
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "head=$(git rev-parse HEAD)"
echo "status_lines=$(git status --porcelain | wc -l)"
git status --porcelain
ls -l "$DST"
md5sum "$DST"
git log --oneline -3
echo "DGXMAIN2 CUT-DONE"
echo "DGXMAIN2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
