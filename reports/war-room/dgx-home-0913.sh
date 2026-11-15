#!/bin/bash
# War-room-sync 09:13 trip: home top level + any npz newer than 08:40 outside the 3 searched trees. READ-ONLY. Marker DGXHOME0913.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-home-0913-ts.log
{
echo "DGXHOME0913 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXHOME0913 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- HOME TOP LEVEL ---'
ls -la /home/arm/junyang/ 2>/dev/null | head -40
echo '--- NPZ NEWER THAN 08:40 OUTSIDE carbon/carbon-repo/s26stage ---'
find /home/arm/junyang -maxdepth 4 -name '*.npz' -newermt '2026-11-14 08:40' 2>/dev/null | grep -v '/home/arm/junyang/carbon' | grep -v '/home/arm/junyang/carbon-repo' | grep -v '/home/arm/junyang/s26stage' | head -10
echo '--- ANY k0_a1_out ANYWHERE UNDER HOME ---'
find /home/arm/junyang -maxdepth 4 -name '*k0_a1_out*' 2>/dev/null | head -5
echo "DGXHOME0913 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXHOME0913 SSH-RC=$?"
echo "DGXHOME0913 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
