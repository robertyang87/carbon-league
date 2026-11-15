#!/bin/bash
# Studio pilot progress probe. READ-ONLY. Marker STUDIOSTAT0742K7M1.
MARK=STUDIOSTAT0742K7M1
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0742-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0742K7M1 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- LAUNCHER + GATE ---'
ps -p 15696,15697 -o pid,etime,args 2>/dev/null | cut -c1-200 || echo LAUNCHER-GATE-GONE
echo '--- WORKERS ---'
pgrep -f spawn_main | wc -l
ps -eo pid,etime,%cpu,args | grep -E 'stage1_teacher_gate|spawn_main' | grep -v grep | cut -c1-200 | head -16
echo '--- PILOT LOG ---'
ls -l /Users/arm/junyang/carbon/data/gen1r_endgame_p1/ | cut -c1-120
wc -l -c /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log
echo '--- PROGRESS LINES (non-PREREG) ---'
grep -v '^PREREG' /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log | cut -c1-300 | tail -15 || echo NO-PROGRESS-LINES
echo '--- LOG TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log | cut -c1-300
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT0742K7M1 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
