#!/bin/bash
# War-room-sync 09:10 trip: locate A0/A1 weight files by mtime (no name filter). READ-ONLY. Marker DGXK0W0910.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0w-0910-ts.log
{
echo "DGXK0W0910 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0W0910 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- ALL NPZ NEWER THAN 08:40 (carbon trees + s26stage) ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo /home/arm/junyang/s26stage -name '*.npz' -newermt '2026-11-14 08:40' 2>/dev/null | head -20
echo '--- WEIGHTS DIR TOP (any new since 08:40?) ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -10
ls -lt /home/arm/junyang/s26stage/data/weights/ 2>/dev/null | head -10
echo '--- K0 LOGS MENTIONING OUT PATH / WEIGHT NAME ---'
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -6
grep -rl 'k0_a1_out' /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -5
grep -rh 'b09e654b\|e5d1e5d2' /home/arm/junyang/carbon/data/dq_logs/ /home/arm/junyang/carbon/data/*.log 2>/dev/null | head -5
echo '--- K0_A1_DS DIR ---'
ls /home/arm/junyang/carbon/data/k0_a1_ds 2>/dev/null | head -5
echo '--- HOLDOUT RECHECK ---'
find /home/arm/junyang/carbon/data/k0_holdout -type f 2>/dev/null | wc -l; ls /home/arm/junyang/carbon/data/k0_holdout/replays 2>/dev/null | wc -l
pgrep -f 'k0_train' | wc -l; pgrep -f 'selfplay_gen' | wc -l
echo "DGXK0W0910 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0W0910 SSH-RC=$?"
echo "DGXK0W0910 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
