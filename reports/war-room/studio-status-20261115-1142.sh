#!/bin/bash
# Studio status probe 2026-11-15 11:42 trip. Read-only.
MARKER=STUDIOSTAT1115-1142
HF=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
LOG="$HF/studio-status-20261115-1142-ts.log"
{
echo "$MARKER PROBE-BEGIN $(date '+%Y-%m-%d %H:%M:%S CST')"
~/workspace/studio_ssh.sh "echo MARKER $MARKER; echo MARKER-IN-STUDIO; hostname; uptime; echo '--- MEM-DISK ---'; df -h /; vm_stat | head -6; echo '--- GATE-PROCS ---'; ps -o pid,etime,%cpu,%mem,command -p \$(pgrep -f 'q_launch.py|stage1_teacher_gate.py' | paste -sd, -); echo '--- WORKERS ---'; pgrep -f 'spawn_mai[n]' | wc -l; ps -o pid,etime,time,%cpu,command -p \$(pgrep -f 'spawn_mai[n]' | paste -sd, -); echo '--- WORKER-CPU-TOTAL ---'; ps -o time= -p \$(pgrep -f 'spawn_mai[n]' | paste -sd, -) | python3 -c 'import sys; t=0.0; n=0
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    parts=line.split(\":\")
    if len(parts)==2: s=float(parts[0])*60+float(parts[1])
    elif len(parts)==3: s=float(parts[0])*3600+float(parts[1])*60+float(parts[2])
    else: s=0.0
    t+=s; n+=1
print(f\"TOTAL_CPU_S={t:.1f} N={n}\")'; echo '--- STAGEB_Q DIR ---'; ls -la /Users/arm/junyang/carbon/data/stageB_q/; ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1; echo '--- Q-LOG-TAIL ---'; tail -c 800 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log; echo; stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S' /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log"
echo "$MARKER SSH-RC=$?"
echo "$MARKER PROBE-END $(date '+%Y-%m-%d %H:%M:%S CST')"
} 2>&1 | tee "$LOG"
