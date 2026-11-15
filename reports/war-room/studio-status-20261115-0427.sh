#!/bin/bash
# Studio status probe 2026-11-15 04:27 trip. Read-only.
MARKER=STUDIOSTAT0427P4R8
HF=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
LOG="$HF/studio-status-20261115-0427-ts.log"
{
echo "$MARKER PROBE-BEGIN $(date '+%Y-%m-%d %H:%M:%S CST')"
~/workspace/studio_ssh.sh "echo MARKER $MARKER; echo MARKER-IN-STUDIO; hostname; uptime; echo '--- MEM-DISK ---'; df -h /; vm_stat | head -6; echo '--- GATE-PROCS ---'; ps -o pid,etime,%cpu,%mem,command -p 72802,72804; echo '--- WORKERS ---'; pgrep -f 'spawn_mai[n]' | wc -l; ps -o pid,etime,time,%cpu,command -p \$(pgrep -f 'spawn_mai[n]' | paste -sd, -); echo '--- STAGEB_Q DIR ---'; ls -la /Users/arm/junyang/carbon/data/stageB_q/; ls /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1; echo '--- Q-LOG-TAIL ---'; tail -c 1200 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log"
echo "$MARKER SSH-RC=$?"
echo "$MARKER PROBE-END $(date '+%Y-%m-%d %H:%M:%S CST')"
} 2>&1 | tee "$LOG"
