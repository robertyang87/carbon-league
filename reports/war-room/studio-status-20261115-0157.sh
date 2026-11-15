#!/bin/bash
# Studio status probe 0157 (2026-11-15 01:57 CST). Last read 01:43: q gate
# run 51972702 RUNNING, launcher 72802 / gate 72804 etime 08:06:43, 20 workers
# ~96-100% CPU each, ~446/1072 roots (41.6%), ~56 roots/h, ETA center ~13:00
# 2026-11-15 (estimate), band 12:50-15:40. Disk 412Gi free.
# Channel: Tailscale SSH direct (arm@100.78.149.20). Task-body channel
# (cloud-assistant -> L20 -> arm@10.211.43.68) is stale; Carbon line avoids
# GitHub Actions (2026-11-14 owner rule).
MARK=STUDIOSTAT0157X9K2
LOG=studio-status-20261115-0157-ts.log
cd ~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/ || exit 1
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
echo "MARKER $MARK"
timeout -k 30 420 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "MARKER-IN-STUDIO"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo '--- GATE-PROCS ---'
ps -p 72802,72804 -o pid,etime,%cpu,%mem,command 2>/dev/null || echo GATE-PIDS-GONE
pgrep -f 'q_du_formal' | head -5
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,%cpu,command | grep -E 'spawn_mai[n]'
echo '--- STAGEB_Q DIR ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ | head -8
echo '--- Q-LOG-TAIL ---'
tail -c 600 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>/dev/null || echo NO-Q-LOG
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
