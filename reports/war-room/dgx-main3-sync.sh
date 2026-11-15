#!/bin/bash
# DGX repo: add main to fetch refspec and ff-sync to origin/main tip. Marker: DGXMAIN3
MARK=DGXMAIN3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXMAIN3 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo "DGXMAIN3 NO-REPO"; exit 1; }
echo '--- PRE-CHECK ---'
BR=$(git rev-parse --abbrev-ref HEAD)
HD=$(git rev-parse HEAD)
ST=$(git status --porcelain)
echo "branch=$BR"
echo "head=$HD"
echo "status_porcelain=[${ST}]"
if [ "$BR" != "main" ]; then echo "DGXMAIN3 ABORT: branch is $BR, expected main"; exit 2; fi
case "$HD" in 5f338eb*) : ;; *) echo "DGXMAIN3 ABORT: HEAD $HD does not start with 5f338eb"; exit 2;; esac
if [ -n "$ST" ]; then echo "DGXMAIN3 ABORT: worktree dirty"; exit 2; fi
echo "DGXMAIN3 PRE-OK"
echo '--- CONFIG-BEFORE ---'
git config --get-all remote.origin.fetch
echo '--- SET-BRANCHES ---'
git remote set-branches --add origin main
echo "set-branches rc=$?"
echo '--- CONFIG-AFTER ---'
git config --get-all remote.origin.fetch
CNT=$(git config --get-all remote.origin.fetch | wc -l)
echo "fetch_refspec_count=$CNT"
echo '--- FETCH ---'
timeout 60 git fetch origin
echo "fetch rc=$?"
echo "origin_main=$(git rev-parse origin/main)"
echo '--- LS-REMOTE ---'
timeout 60 git ls-remote origin main
echo "ls-remote rc=$?"
echo '--- FF-MERGE ---'
git merge --ff-only origin/main
echo "ff-merge rc=$?"
echo '=== ATTEST ==='
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
echo "head=$(git rev-parse HEAD)"
echo "origin_main=$(git rev-parse origin/main)"
echo "status_lines=$(git status --porcelain | wc -l)"
git status --porcelain
ls -l data/m9xj320_matches_101.json
python3 -c "import json; print('fable_len=', len(json.load(open('data/fable_matches.json'))))"
python3 -c "import json; print('m9xj320_len=', len(json.load(open('data/m9xj320_matches.json'))))"
git log --oneline -4
echo "DGXMAIN3 SYNC-DONE"
echo "DGXMAIN3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
