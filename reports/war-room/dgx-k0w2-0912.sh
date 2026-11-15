#!/bin/bash
# War-room-sync 09:12 trip: broad any-file mtime sweep for A0/A1 outputs. READ-ONLY. Marker DGXK0W20912.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0w2-0912-ts.log
{
echo "DGXK0W20912 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0W20912 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- HOME TOP LEVEL ---'
ls /home/arm/junyang/ 2>/dev/null
echo '--- ANY FILE NEWER THAN 08:40, carbon trees (excl k0_holdout + k0_a1 replays) ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -type f -newermt '2026-11-14 08:40' 2>/dev/null | grep -v 'k0_holdout' | grep -v 'data/k0_a1/replays' | head -30
echo '--- COUNT OF ABOVE ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -type f -newermt '2026-11-14 08:40' 2>/dev/null | grep -v 'k0_holdout' | grep -v 'data/k0_a1/replays' | wc -l
echo '--- S26STAGE ANY FILE NEWER THAN 08:40 ---'
find /home/arm/junyang/s26stage -type f -newermt '2026-11-14 08:40' 2>/dev/null | head -10
echo '--- TMP / ROOT CARBON ---'
ls -lt /tmp/ 2>/dev/null | head -5
ls /root/carbon 2>/dev/null | head -5
echo "DGXK0W20912 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0W20912 SSH-RC=$?"
echo "DGXK0W20912 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
