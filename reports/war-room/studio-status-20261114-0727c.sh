#!/bin/bash
# Studio pilot acceptance: log head + progress lines + stray rl2 check. READ-ONLY. Marker STUDIOSTAT0727Q9X4C.
MARK=STUDIOSTAT0727Q9X4C
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0727c-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0727Q9X4C STUDIO-BEGIN $(date '+%F %T %Z')"
wc -l -c /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log
head -c 1200 /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log; echo
echo '--- PROGRESS LINES (non-PREREG) ---'
grep -v '^PREREG' /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log | cut -c1-300 | tail -10 || echo NO-PROGRESS-LINES
echo '--- RL2 15768 STILL THERE? ---'
ps -p 15768 -o pid,etime,args 2>/dev/null | cut -c1-200 || echo PID15768-GONE
ps -p 15753 -o pid,etime,args 2>/dev/null | cut -c1-200 || echo PID15753-GONE
echo '--- LAUNCHER + GATE STILL UP? ---'
ps -p 15696,15697 -o pid,etime,args | cut -c1-160
echo '--- VERIFY SCRIPT OUTPUT? ---'
ls -l /Users/arm/junyang/carbon/data/gen1r_endgame_p1/*.out /Users/arm/junyang/carbon/data/gen1r_endgame_p1/*verify* 2>/dev/null
grep -riE '119600|mirror|cap2|400' /Users/arm/junyang/carbon/data/gen1r_endgame_p1/p1_launch.out /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log 2>/dev/null | cut -c1-200 | head -5 || echo NO-VERIFY-NUMBERS-IN-LOGS
echo "STUDIOSTAT0727Q9X4C STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
