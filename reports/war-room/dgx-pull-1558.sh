#!/bin/bash
# DGX pull retry trip 2026-11-13 15:58 - per war-room ruling: retry fetch ONCE; if it fails,
# run channel diagnosis (curl github.com / git ls-remote), no hard re-retry.
# If fetch succeeds, complete the standing 15:47 pull order: stash bigeval.jsonl (if dirty),
# ff-merge to 5aa7b544, stash pop, verify HEAD + P1 md5s + bigeval status.
# Channel: Tailscale SSH arm@100.84.148.73. Marker DGXPULL1558.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXPULL1558 $(date '+%H:%M:%S %Z')"
timeout 300 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1558-PULL; cd /home/arm/junyang/carbon-repo || exit 1
echo ---PRE-STATUS---; git rev-parse HEAD; git status --porcelain -- data/bigeval.jsonl; echo "---STASH-STEP---"
STASHED=0
if [ -n "$(git status --porcelain -- data/bigeval.jsonl)" ]; then git stash push -- data/bigeval.jsonl && STASHED=1; echo "STASHED=1"; else echo "STASHED=0 (bigeval clean)"; fi
echo ---FETCH-RETRY-ONCE---; timeout 120 git fetch origin main; FETCH_RC=$?; echo "FETCH_RC=$FETCH_RC"
if [ $FETCH_RC -eq 0 ]; then
  echo ---FETCH-OK-VERIFY-OBJECT---; git cat-file -t 5aa7b54415d4361e0f7d7278aa4f480ea9aa586a; git rev-parse origin/main
  echo ---FF-MERGE-TO-5aa7b544---; timeout 60 git merge --ff-only 5aa7b54415d4361e0f7d7278aa4f480ea9aa586a; MERGE_RC=$?; echo "MERGE_RC=$MERGE_RC"
  if [ $STASHED -eq 1 ]; then echo ---STASH-POP---; git stash pop; echo "POP_RC=$?"; fi
  echo ---POST-VERIFY---; git rev-parse HEAD; git status --porcelain -- data/bigeval.jsonl; echo "(empty above = worktree clean for bigeval)"; md5sum code/cnet2_build.py code/cnet2_train.py code/set_uw.py
else
  echo ---FETCH-FAILED-DIAGNOSIS---; echo "curl github.com:"; timeout 20 curl -s -o /dev/null -w "HTTP %{http_code} time %{time_total}s\n" https://github.com || echo "CURL_RC=$?"
  echo "git ls-remote origin main:"; timeout 60 git ls-remote origin main; echo "LSREMOTE_RC=$?"
  if [ $STASHED -eq 1 ]; then echo ---STASH-POP-RESTORE---; git stash pop; echo "POP_RC=$?"; fi
  echo ---POST-FAIL-STATUS---; git rev-parse HEAD; git status --porcelain -- data/bigeval.jsonl
fi
echo ---END-PULL---' 2>&1
echo "rcPULL=$?"
echo "DGXPULL1558-END $(date '+%H:%M:%S %Z')"
