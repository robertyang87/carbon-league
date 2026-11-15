#!/bin/bash
# War-room-sync 09:16 trip: md5 the A0/A1 checkpoint FILES (out is a file, not a dir). READ-ONLY. Marker DGXK0M0916.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0md5-0916-ts.log
{
echo "DGXK0M0916 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0M0916 DGX-BEGIN $(date '+%F %T %Z')"
ls -l /home/arm/junyang/k0_a0_out /home/arm/junyang/k0_a1_out /home/arm/junyang/k0_a0_out.complete.json /home/arm/junyang/k0_a1_out.complete.json /home/arm/junyang/k0_a0_out.log.jsonl /home/arm/junyang/k0_a1_out.log.jsonl 2>&1
echo '--- MD5 (receipt expects A0 b09e654b / A1 e5d1e5d2) ---'
md5sum /home/arm/junyang/k0_a0_out /home/arm/junyang/k0_a1_out 2>&1
echo '--- SHA256 cross-check vs complete.json ---'
sha256sum /home/arm/junyang/k0_a0_out /home/arm/junyang/k0_a1_out 2>&1
echo '--- B0 + HOLDOUT STATUS ---'
pgrep -f 'k0_train' | wc -l; pgrep -f 'selfplay_gen' | wc -l
ls /home/arm/junyang/carbon/data/k0_holdout/replays 2>/dev/null | wc -l
echo "DGXK0M0916 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0M0916 SSH-RC=$?"
echo "DGXK0M0916 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
