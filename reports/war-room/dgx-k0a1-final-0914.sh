#!/bin/bash
# War-room-sync 09:14 trip: verify A0/A1 weights at /home/arm/junyang/k0_a1_out. READ-ONLY. Marker DGXK0F0914.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0a1-final-0914-ts.log
{
echo "DGXK0F0914 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0F0914 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K0_A1_OUT LISTING ---'
ls -lt /home/arm/junyang/k0_a1_out/ 2>/dev/null
echo '--- COMPLETE JSON ---'
cat /home/arm/junyang/k0_a1_out.complete.json 2>/dev/null | head -20
echo '--- LOG JSONL TAIL ---'
tail -5 /home/arm/junyang/k0_a1_out.log.jsonl 2>/dev/null | cut -c1-400
echo '--- MD5 OF WEIGHT FILES (expect A0 b09e654b / A1 e5d1e5d2) ---'
md5sum /home/arm/junyang/k0_a1_out/*.npz 2>/dev/null
echo '--- B0 STATUS ---'
pgrep -f 'k0_train' | wc -l; pgrep -f 'selfplay_gen' | wc -l
ps -eo pid,etime,args | grep '[k]0_train' | cut -c1-200 || echo NO-K0-TRAIN
echo "DGXK0F0914 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0F0914 SSH-RC=$?"
echo "DGXK0F0914 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
