#!/bin/bash
# War-room-sync 09:26 trip: DGX K0 B0/packs + clone status. READ-ONLY. Marker DGXK00926.
MARK=DGXK00926
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0-0926-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK00926 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K0 TRAIN PROCS ---'
ps -eo pid,etime,args | grep '[k]0_train' | cut -c1-220 || echo NO-K0-TRAIN
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1
echo '--- B0/A0/A1 OUTS ---'
ls -l /home/arm/junyang/k0_b0_out /home/arm/junyang/k0_a0_out /home/arm/junyang/k0_a1_out 2>&1
md5sum /home/arm/junyang/k0_b0_out 2>/dev/null
echo '--- PACKS IN CLONE ---'
ls -l /home/arm/junyang/carbon-repo/code/exp/k0_*.py 2>&1
md5sum /home/arm/junyang/carbon-repo/code/exp/k0_*.py 2>/dev/null
echo '--- CLONE HEAD ---'
git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD 2>&1; git -C /home/arm/junyang/carbon-repo status --porcelain 2>&1 | head -5
echo '--- ANY NEW FILES SINCE 09:20 (carbon trees) ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -type f -newermt '2026-11-14 09:20' 2>/dev/null | grep -v k0_holdout | head -20
echo '--- HOLDOUT DIR ---'
ls /home/arm/junyang/carbon/data/k0_holdout 2>/dev/null | head -3; ls /home/arm/junyang/carbon/data/k0_holdout/replays 2>/dev/null | wc -l
echo "DGXK00926 DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
