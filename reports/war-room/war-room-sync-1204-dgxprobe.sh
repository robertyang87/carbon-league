#!/bin/bash
# War-room-sync 12:04 trip (v136): DGX duel cell-3 + Echo4 pack status. READ-ONLY. Marker DGXSYNC1204.
MARK=DGXSYNC1204
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1204-dgxprobe.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSYNC1204 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1
echo '--- DUEL-PROCS ---'
ps -eo pid,etime,%cpu,args | grep -E '[k]0_duel|[b]igeval' | cut -c1-200 || echo NO-DUEL-PROC
echo '--- DUEL-DIR ---'
ls -la --time-style=long-iso /home/arm/junyang/k0_calib/duels/ 2>&1 | tail -8
echo '--- ECHO4-PACK ---'
ls -l /home/arm/junyang/carbon-repo/code/packs/echo4_int8.py /home/arm/junyang/carbon/code/packs/echo4_int8.py 2>&1
md5sum /home/arm/junyang/carbon-repo/code/packs/echo4_int8.py /home/arm/junyang/carbon/code/packs/echo4_int8.py 2>&1
echo '--- ECHO4-WEIGHT-COPY ---'
ls -l /home/arm/junyang/*ckpt_20* /home/arm/junyang/carbon*/data/weights/*ppou2* 2>&1 | head -10
find /home/arm/junyang -maxdepth 4 -name '*echo4*' -newer /home/arm/junyang/k0_calib 2>/dev/null | head -10
echo '--- ECHO4-LOGS ---'
ls -lt --time-style=long-iso /home/arm/junyang/*.log 2>/dev/null | head -5
ls -lt --time-style=long-iso /home/arm/junyang/carbon-repo/*.log 2>/dev/null | head -5
echo '--- REPO-HEAD ---'
git -C /home/arm/junyang/carbon-repo rev-parse HEAD 2>&1 | head -2
echo "DGXSYNC1204 DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
