#!/bin/bash
# Studio status probe 2026-11-14 16:12 (cron studio-status-report, dated name to avoid
# stale 2026-11-12 studio-status-1612.sh Fable-era file).
# Context: m7bsr1 A4 DONE 15:57; q-gate dispatch 16:04 (war room, sole executor Studio);
# 16:09 probe found NO gate process. Receipt file q-gate-start-receipt-20261114.md
# claims gate started 16:11:24 (launcher 51489, gate 51491, 12 workers, PREREG landed).
# This trip: INDEPENDENT direct-read verification of the q-gate start three-piece.
MARK=STUDIOSTAT1612C
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1612-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 150 ~/workspace/studio_ssh.sh "bash -s" <<'STEOF'
echo "STUDIOSTAT1612C STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GATE-LAUNCHER-51489 ---'
ps -p 51489 -o pid,etime,cputime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID51489-GONE
echo '--- GATE-PROC-51491 ---'
ps -p 51491 -o pid,etime,cputime,%cpu,args 2>/dev/null | cut -c1-220 || echo PID51491-GONE
echo '--- GATE-ANY (stage1_teacher_gate / q_launch) ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'stage1_teacher_gate|q_launch|stage1|teacher_gate' | grep -v grep | cut -c1-220 || echo NO-GATE-PROC
echo '--- SPAWN_MAIN COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- LOAD ---'
uptime
echo '--- STAGEB_Q DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ 2>/dev/null | head -10 || echo NO-STAGEB_Q-DIR
echo '--- Q_DU_FORMAL LOG ---'
f=/Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log
if [ -f "$f" ]; then ls -l "$f"; tail -6 "$f" | cut -c1-300; else echo NO-Q-DU-FORMAL-LOG; fi
echo '--- A4-RESIDUAL ---'
ps -ax -o pid,etime,command | grep -E '[a]4_select|[c]net2_train' | cut -c1-200 | head -5 || echo NO-A4-RESIDUAL
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1612C STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "probe done rc=$? log=$LOG"
