#!/bin/bash
# War-room-sync 02:39 trip (v98): DGX k0 duel liveness check via Tailscale SSH direct. READ-ONLY. Marker DGXSTATUS0239.
MARK=DGXSTATUS0239
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0239-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSTATUS0239 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K0-DUEL-PROCS ---'
ps -eo pid,etime,%cpu,args | grep '[k]0_duel' | cut -c1-220 | head -25
echo '--- DUEL-WORKER-COUNT ---'
pgrep -f 'k0_duel' | wc -l
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1 | head -3
echo '--- DUELS-DIR ---'
ls -la --time-style=+%F_%T /home/arm/junyang/k0_calib/duels/ 2>&1 | head -10
echo '--- DRIVER-LOG-TAIL ---'
tail -5 /home/arm/junyang/k0_calib/driver.log 2>&1 | cut -c1-250
echo '--- DUEL-LOG ---'
ls -l --time-style=+%F_%T /home/arm/junyang/k0_calib/duel_lr0.0002_s100_b1_seed7.log 2>&1
echo "DGXSTATUS0239 DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
