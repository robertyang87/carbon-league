#!/bin/bash
# War-room-sync 09:05 trip follow-up: locate A0/A1 weights + holdout progress. READ-ONLY. Marker DGXK0V0905.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0a1-verify-0905-ts.log
{
echo "DGXK0V0905 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0V0905 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- FIND k0_a1_out DIRS ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 3 -name '*k0_a1_out*' 2>/dev/null
echo '--- LS CANDIDATE OUT DIRS ---'
ls -lt /home/arm/junyang/carbon/k0_a1_out 2>/dev/null | head -8
ls -lt /home/arm/junyang/carbon/data/k0_a1 2>/dev/null | head -8
echo '--- A0/A1 WEIGHT FILES (new npz since 08:40, k0 dirs) ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -name '*.npz' -newermt '2026-11-14 08:40' 2>/dev/null | grep -i 'k0\|a0\|a1' | head -10
find /home/arm/junyang/carbon/data/k0_a1_out /home/arm/junyang/carbon/k0_a1_out /home/arm/junyang/carbon/data/k0_a1 -type f 2>/dev/null | head -10
echo '--- MD5 OF A0/A1 CANDIDATES (expect b09e654b / e5d1e5d2 prefixes) ---'
for f in /home/arm/junyang/carbon/k0_a1_out/* /home/arm/junyang/carbon/data/k0_a1_out/* /home/arm/junyang/carbon/data/k0_a1/*; do [ -f "$f" ] && md5sum "$f"; done 2>/dev/null | head -10
echo '--- HOLDOUT PROGRESS ---'
ls /home/arm/junyang/carbon/data/k0_holdout 2>/dev/null | wc -l
ls -lt /home/arm/junyang/carbon/data/k0_holdout 2>/dev/null | head -4
find /home/arm/junyang/carbon/data/k0_holdout -type f 2>/dev/null | wc -l
echo '--- K0_TRAIN PROC (B0 started?) ---'
pgrep -f 'k0_train' | wc -l
ps -eo pid,etime,args | grep '[k]0_train' | cut -c1-200 || echo NO-K0-TRAIN
echo "DGXK0V0905 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0V0905 SSH-RC=$?"
echo "DGXK0V0905 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
