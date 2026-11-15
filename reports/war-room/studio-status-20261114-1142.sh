#!/bin/bash
# Studio status report 11:42 trip: A4 g1 selection progress check. READ-ONLY. Marker STUDIOSTAT1142.
MARK=STUDIOSTAT1142
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1142-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1142 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- A4 DRIVER PROCS ---'
ps -eo pid,etime,%cpu,args | grep -E 'a4_select|spawn_mai[n]|cnet2_train' | grep -v grep | cut -c1-220 | head -20 || echo NO-A4-PROCS
echo '--- SPAWN COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- TOP CPU ---'
ps -eo pid,%cpu,etime,args | sort -k2 -rn | head -5 | cut -c1-200
echo '--- A4 SELECTION DIR ---'
ls -lt /Users/arm/junyang/a4_selection_20261114/ | cut -c1-160 | head -20
echo '--- G1 SEL DIR ---'
ls -lt /Users/arm/junyang/a4_selection_20261114/g1_sel/ 2>/dev/null | cut -c1-160 | head -25
echo '--- G1 META (if present) ---'
cat /Users/arm/junyang/a4_selection_20261114/g1_selection.json 2>/dev/null | head -60 || echo NO-G1-META-YET
echo '--- G1 LOG TAIL ---'
ls -lt /Users/arm/junyang/a4_selection_20261114/*.log /Users/arm/junyang/a4_selection_20261114/g1_sel/*.log 2>/dev/null | cut -c1-160 | head
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1142 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
