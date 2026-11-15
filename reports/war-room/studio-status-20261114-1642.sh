#!/bin/bash
# Studio status probe 2026-11-14 16:42 (cron studio-status-report, dated name).
# Context: formal q gate running since 16:11:24 (launcher 51489, gate 51491,
# 12 workers, PREREG run c132bf81, 1072 roots / 323 games, ETA ~11-15 22:00).
# 16:40 probe (war-room sync): etime 29:11, load 12.95, q_du_formal.json not out.
# This trip: independent direct-read verification that the gate is still
# progressing (etime + worker cputime advancing, load, fresh files).
MARK=STUDIOSTAT1642C
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1642-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 150 ~/workspace/studio_ssh.sh "bash -s" <<'STEOF'
echo "STUDIOSTAT1642C STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GATE-LAUNCHER-51489 ---'
ps -p 51489 -o pid,etime,time,%cpu,args 2>/dev/null | cut -c1-200 || echo PID51489-GONE
echo '--- GATE-PROC-51491 ---'
ps -p 51491 -o pid,etime,time,%cpu,args 2>/dev/null | cut -c1-200 || echo PID51491-GONE
echo '--- GATE-ANY (stage1_teacher_gate / q_launch) ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'stage1_teacher_gate|q_launch' | grep -v grep | cut -c1-200 || echo NO-GATE-PROC
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -ax -o pid,etime,time,%cpu,command | grep 'spawn_mai[n]' | awk '{print $1, $2, $3, $4}' | head -14
echo '--- LOAD ---'
uptime
echo '--- STAGEB_Q DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ 2>/dev/null | head -12 || echo NO-STAGEB_Q-DIR
echo '--- FRESH FILES (stageB_q, mmin -20) ---'
find /Users/arm/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head -10; echo FRESH-DONE
echo '--- Q_DU_FORMAL OUTPUT ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>/dev/null || echo NO-Q_DU_FORMAL-JSON
f=/Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log
if [ -f "$f" ]; then ls -l "$f"; tail -3 "$f" | cut -c1-300; else echo NO-Q-DU-FORMAL-LOG; fi
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1642C STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "probe done rc=$? log=$LOG"
