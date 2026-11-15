#!/bin/bash
# Probe A: DGX repo fetch + ff-only to origin/main, read back HEAD, verify B-group commits are ancestors.
# Executor: DGX side chat (sole executor per war-room dispatch 2026-11-12 23:14). Marker: PBSYNC2314A
MARK=PBSYNC2314A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBSYNC2314A DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo '--- SELF-SCAN GIT PROCS ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout|merge)' | grep -v grep || echo NO-GIT-PROC
ls -l /home/arm/junyang/carbon-repo/.git/index.lock 2>/dev/null || echo NO-INDEX-LOCK
echo '--- REPO STATE BEFORE ---'
cd /home/arm/junyang/carbon-repo || exit 9
echo -n "branch="; timeout 30 git rev-parse --abbrev-ref HEAD
echo -n "head-before="; timeout 30 git rev-parse HEAD
echo '--- STATUS (tracked changes would block ff) ---'
timeout 30 git status --porcelain=v1 | head -10; echo "status-lines=$(timeout 30 git status --porcelain=v1 | wc -l)"
echo '--- FETCH ---'
timeout 120 git fetch origin main; echo "fetch-rc=$?"
echo -n "origin-main="; timeout 30 git rev-parse origin/main
echo '--- FF-ONLY MERGE ---'
timeout 60 git merge --ff-only origin/main; echo "ff-rc=$?"
echo '--- REPO STATE AFTER ---'
echo -n "head-after="; timeout 30 git rev-parse HEAD
echo -n "branch-after="; timeout 30 git rev-parse --abbrev-ref HEAD
timeout 30 git log --oneline -5
echo '--- B-GROUP ANCESTORS ---'
for sha in 6facdfe2 ddc39a57 24957e17 d87b8ee6; do
  if timeout 30 git merge-base --is-ancestor $sha HEAD 2>/dev/null; then echo "$sha IN"; else echo "$sha MISSING"; fi
done
echo '--- ROLLOUT FILE REPO vs WORKSPACE ---'
md5sum /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py /home/arm/junyang/carbon/code/rl2_ppo_rollout.py 2>/dev/null
ls -l --time-style=+%F_%T /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py /home/arm/junyang/carbon/code/rl2_ppo_rollout.py 2>/dev/null
echo '--- PLAY_BATCH SIGNATURE (repo, new code) ---'
grep -n "def play_batch" -A 12 /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py | head -30
echo '--- PLAY_BATCH CALLERS ---'
grep -rn "play_batch" /home/arm/junyang/carbon-repo/code/*.py | grep -v "def play_batch" | head -20
echo "PBSYNC2314A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
