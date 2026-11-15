#!/bin/bash
# DGX status report 04:58 trip, follow-up probe C: spot-32 deployment markers + full tailnet + repo origin. Marker DGXSTATUS0458C.
MARK=DGXSTATUS0458C
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0458c-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSTATUS0458C DGX-BEGIN $(date '+%F %T %Z')"
echo '--- TAILSCALE-FULL ---'
tailscale status 2>&1 | head -25
echo '--- SPOT32-MARKERS ---'
find /home/arm/junyang -maxdepth 3 \( -iname '*spot*32*' -o -iname '*spot32*' -o -iname '*exam*ready*' -o -iname '*EXAM-READY*' \) 2>/dev/null | head -20
echo '--- REPO-ORIGIN ---'
git -C /home/arm/junyang/carbon-repo remote -v 2>&1 | head -4
echo '--- SPOT32-CONN ---'
ps -eo pid,etime,args | grep -E '[1]00\.95\.45\.69|[s]pot-32' | head -5; echo NO_SPOT32_PROC
echo "DGXSTATUS0458C DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
