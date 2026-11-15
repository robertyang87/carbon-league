#!/bin/bash
# Probe B: read back post-ff HEAD, B-group ancestors, rollout md5 repo vs workspace, play_batch signature/callers.
# Read-only. Marker: PBSYNC2314B
MARK=PBSYNC2314B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBSYNC2314B DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || exit 9
echo -n "head-after="; timeout 30 git rev-parse HEAD
echo -n "branch="; timeout 30 git rev-parse --abbrev-ref HEAD
echo -n "origin-main="; timeout 30 git rev-parse origin/main
echo -n "behind="; timeout 30 git rev-list --count HEAD..origin/main
timeout 30 git log --oneline -5
echo '--- B-GROUP ANCESTORS ---'
for sha in 6facdfe2 ddc39a57 24957e17 d87b8ee6; do
  if timeout 30 git merge-base --is-ancestor $sha HEAD 2>/dev/null; then echo "$sha IN"; else echo "$sha MISSING"; fi
done
echo '--- ROLLOUT MD5 ---'
md5sum /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py /home/arm/junyang/carbon/code/rl2_ppo_rollout.py 2>/dev/null || echo MD5-FAIL
echo '--- PLAY_BATCH SIGNATURE ---'
grep -n "def play_batch" -A 12 /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py | head -30
echo '--- PLAY_BATCH CALLERS ---'
grep -rn "play_batch" /home/arm/junyang/carbon-repo/code/*.py | grep -v "def play_batch" | head -20
echo '--- ROLLOUT IMPORTS/TOP ---'
head -40 /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py
echo "PBSYNC2314B DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
