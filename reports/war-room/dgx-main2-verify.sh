#!/bin/bash
# DGX repo cut to main, trip 2 step 1: READ-ONLY pre-check. Marker: DGXMAIN2
MARK=DGXMAIN2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN2 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN2 NO-REPO"; exit 1; }
echo '--- BRANCH ---'
git rev-parse --abbrev-ref HEAD
echo '--- HEAD ---'
git rev-parse HEAD
echo '--- STATUS-PORCELAIN ---'
git status --porcelain
echo '--- FILE ---'
ls -l code/matches_d679ed45ab595e8b7bb1ea51c3451154.json
md5sum code/matches_d679ed45ab595e8b7bb1ea51c3451154.json
echo '--- REMOTE-MAIN ---'
timeout 60 git ls-remote origin main
echo "lsremote rc=$?"
echo "DGXMAIN2 VERIFY-DONE"
echo "DGXMAIN2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
