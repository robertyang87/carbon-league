#!/bin/bash
# Studio status probe 2026-11-15 06:57 trip. Read-only.
MARKER=STUDIOSTAT0657K3M9
HF=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
LOG="$HF/studio-status-20261115-0657-ts.log"
{
echo "$MARKER PROBE-BEGIN $(date '+%Y-%m-%d %H:%M:%S CST')"
~/workspace/studio_ssh.sh "echo MARKER $MARKER; echo MARKER-IN-STUDIO; hostname; uptime; echo '--- MEM-DISK ---'; df -h /; vm_stat | head -6; echo '--- GATE-PROCS ---'; ps -o pid,etime,%cpu,%mem,command -p \$(pgrep -f 'q_launch.py|stage1_teacher_gate.py' | paste -sd, -); echo '--- WORKERS ---'; pgrep -f 'spawn_mai[n]' | wc -l; ps -o pid,etime,time,%cpu,command -p \$(pgrep -f 'spawn_mai[n]' | paste -sd, -); echo '--- WORKER-CPU-TOTAL ---'; ps -o time= -p \$(pgrep -f 'spawn_mai[n]' | paste -sd, -) | awk -F: '{s+=\$1*3600+\$2*60+\$3} END {print \"TOTAL_CPU_S=\" s \" N=\" NR}'; echo '--- STAGEB_Q DIR ---'; ls -la /Users/arm/junyang/carbon/data/stageB_q/; ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1; echo '--- Q-LOG-TAIL ---'; tail -c 800 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log; echo; stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S' /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log"
echo "$MARKER SSH-RC=$?"
echo "$MARKER PROBE-END $(date '+%Y-%m-%d %H:%M:%S CST')"
} 2>&1 | tee "$LOG"
