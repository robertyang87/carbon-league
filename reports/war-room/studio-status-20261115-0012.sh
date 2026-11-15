#!/bin/bash
# Studio status probe 0012 (2026-11-15 00:12 CST). Last read 23:59: q gate
# run 51972702 RUNNING, launcher 72802 / gate 72804 etime 06:22:52, 20 workers
# ~98% CPU each, ~350/1072 roots (32.7%), ~56 roots/h, ETA center ~13:20-13:50
# 2026-11-15 (estimate), band 12:50-15:40. Disk 412Gi free.
# Channel: Tailscale SSH direct (arm@100.78.149.20), per snapshot note.
MARK=STUDIOSTAT0012K7Q3
LOG=studio-status-20261115-0012-ts.log
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
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,%cpu,command | grep -E 'spawn_mai[n]' | head -5
echo '--- STAGEB_Q DIR ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ | head -8
echo '--- Q-LOG-TAIL ---'
tail -3 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>/dev/null || echo NO-Q-LOG
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
