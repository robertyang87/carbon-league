#!/bin/bash
# Studio status report 14:57 trip. READ-ONLY. Marker STUDIOSTAT1115-1457.
MARK=STUDIOSTAT1115-1457
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261115-1457-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1115-1457 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- LAUNCHER-GATE ---'
ps -p 72802,72804 -o pid,etime,%cpu,%mem,args 2>/dev/null || echo NO-72802-72804
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,args | grep '[s]pawn_main' | awk '{s+=$3; n++} END {print "WORKER_CPU_TOTAL_S=" s " N=" n}'
echo '--- LOAD/MEM/DISK ---'
vm_stat | head -6
df -h / | tail -1
echo '--- STAGEB_Q DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ | head -15
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1
ls -l /Users/arm/junyang/carbon/data/stageB_q/*.progress.json /Users/arm/junyang/carbon/data/stageB_q/*GATE-RESULT* 2>&1 | head -5
echo '--- Q-LOG ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log
tail -c 600 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log
echo
echo "STUDIOSTAT1115-1457 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
