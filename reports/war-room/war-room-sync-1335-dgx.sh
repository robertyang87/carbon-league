#!/bin/bash
MARK=DGXSYNC1335
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1335-dgx.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSYNC1335 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1
echo '--- distill train proc ---'
ps -eo pid,etime,%cpu,args | grep -E '[t]rain_lr0|lr0.0002_s400' | cut -c1-160
echo '--- distill log tail ---'
tail -3 /home/arm/junyang/k0_calib/train_lr0.0002_s400_b1_seed7.log 2>&1
ls -la --time-style=long-iso /home/arm/junyang/k0_calib/cells/lr0.0002_s400_b1_seed7.npz 2>&1
echo '--- gate2 drivers (katago_probe_scratch) ---'
ps -eo pid,etime,%cpu,args | grep '[g]ate2_driver' | wc -l
ps -eo pid,etime,%cpu,args | grep '[g]ate2_driver' | head -4 | cut -c1-160
ls -la --time-style=long-iso /home/arm/junyang/carbon/katago_probe_scratch/gatesA/ 2>&1 | head -15
echo '--- duel cell latest ---'
ls -lt /home/arm/junyang/k0_calib/duels/*.json 2>&1 | head -3
echo "DGXSYNC1335 DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
