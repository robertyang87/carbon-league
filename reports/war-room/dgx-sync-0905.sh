#!/bin/bash
# War-room-sync 09:05 trip: DGX B0/holdout/packaging receipts + A0/A1 weights. READ-ONLY. Marker DGXSYNC0905.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-sync-0905-ts.log
{
echo "DGXSYNC0905 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSYNC0905 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- GIT HEAD (carbon-repo) ---'
cd /home/arm/junyang/carbon-repo && git rev-parse --short HEAD 2>&1; git log --oneline -2 2>&1 | cut -c1-120
echo '--- K0 TRAIN / SELFPLAY PROCS ---'
pgrep -f 'k0_train' | wc -l; pgrep -f 'selfplay_gen' | wc -l
ps -eo pid,ppid,etime,args | grep -E '[k]0_train|[s]elfplay_gen' | cut -c1-250 || echo NO-K0-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv 2>&1
echo '--- K0_TRAIN MD5 (main new 6e0bc68d after Codex lr fix; old 0f464b2f) ---'
md5sum /home/arm/junyang/carbon-repo/code/k0_train.py 2>/dev/null; md5sum /home/arm/junyang/carbon/code/k0_train.py 2>/dev/null
echo '--- K0_A1 OUT (A0 detach + A1 backbone weights) ---'
ls -lt /home/arm/junyang/carbon/data/k0_a1_out 2>/dev/null | head -8; ls /home/arm/junyang/carbon/data/k0_a1_out 2>/dev/null | wc -l
echo '--- B0 / HOLDOUT / PACKAGING (new since 08:55) ---'
find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/data -maxdepth 3 \( -iname '*b0*' -o -iname '*holdout*' -o -iname '*9870*' -o -iname '*screen*' -o -iname '*k0*' \) -newermt '2026-11-14 08:55' 2>/dev/null | head -20
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -6
echo '--- LOAD ---'
uptime
echo "DGXSYNC0905 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXSYNC0905 SSH-RC=$?"
echo "DGXSYNC0905 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
