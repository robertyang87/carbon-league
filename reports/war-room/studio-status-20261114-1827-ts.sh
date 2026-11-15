#!/bin/bash
# Studio status probe 2026-11-14 18:27 (fresh, read-only, Tailscale direct).
# Marker: STUDIOSTAT1827X7K3 (claim only output containing this marker)
MARK=STUDIOSTAT1827X7K3
LOG="$HOME/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1827-ts.log"
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ~/workspace/studio_ssh.sh 'bash -s' <<'STEOF'
echo "STUDIOSTAT1827X7K3 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- GATE-PROCS ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E '72802|72804|q_launch|gate|du_formal' | grep -v grep || echo NO-GATE-MATCH
echo '--- WORKERS ---'
ps -ax -o pid,%cpu,etime,cputime,command | grep 'spawn_main' | grep -v grep | wc -l
ps -ax -o pid,%cpu,etime,cputime,command | grep 'spawn_main' | grep -v grep | head -3
ps -ax -o pid,%cpu,etime,cputime,command | grep 'spawn_main' | grep -v grep | tail -3
echo '--- OLD-RUN-RESIDUE ---'
ps -ax -o pid,command | grep -E '51489|51491' | grep -v grep || echo NO-OLD-RUN
echo '--- LOAD ---'
uptime
echo '--- STAGEB_Q DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ | head -8
echo '--- FRESH FILES (stageB_q, mmin -20) ---'
find /Users/arm/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head -5; echo FRESH-DONE
echo '--- Q_DU_FORMAL OUTPUT ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>/dev/null || echo NO-Q_DU_FORMAL-JSON
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log
echo '--- WEIGHTS (latest 5) ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -6
echo "STUDIOSTAT1827X7K3 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
