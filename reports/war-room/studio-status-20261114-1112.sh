#!/bin/bash
# Studio status report 10:57 trip: P1 B segment progress check. READ-ONLY. Marker STUDIOSTAT1112.
MARK=STUDIOSTAT1112
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1112-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1112 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- LAUNCHER + B GATE ---'
ps -p 15696,59060 -o pid,etime,args 2>/dev/null | cut -c1-200 || echo LAUNCHER-BGATE-GONE
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,%cpu,args | grep -E 'stage1_teacher_gate|spawn_mai[n]' | grep -v grep | cut -c1-200 | head -16
echo '--- PILOT DIR ---'
ls -l /Users/arm/junyang/carbon/data/gen1r_endgame_p1/ | cut -c1-120
echo '--- B LOG ---'
wc -l -c /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1b.log
echo '--- B PROGRESS LINES (non-PREREG) ---'
grep -v '^PREREG' /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1b.log | cut -c1-300 | tail -15 || echo NO-PROGRESS-LINES
echo '--- B LOG TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1b.log | cut -c1-300
echo '--- RESULT FILES (new since 11:00) ---'
find /Users/arm/junyang/carbon/data/gen1r_endgame_p1 -type f -newermt '2026-11-14 11:00' -exec ls -l {} \; | cut -c1-160
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1112 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
