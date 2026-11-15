#!/bin/bash
# Studio status probe 2026-11-14 16:10: formal q-gate start receipt check.
# Context: q-gate released (gate e922d9d8, test fd4cf144, seed-cont 995400, grade formal,
# wall est ~29.8h), dispatch sent to Studio (submission e9d1f5f0), A4 m7bsr1 done 15:57.
# This trip: any q-gate process / new gate launcher log / stage1_formal activity?
MARK=STUDIOSTAT1610
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-1610.log
{
echo "$MARK BEGIN $(date '+%F %T %Z')"
~/workspace/studio_ssh.sh "bash -s" <<'STEOF'
echo "STUDIOSTAT1610 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GATE-TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'stage1|teacher_gate|cnet2_train|k0_train|a4_select' | grep -v grep || echo NO-GATE-TRAIN-PROC
echo '--- OTHER-PYTHON-PROC ---'
ps -ax -o pid,%cpu,etime,command | grep -E '[p]ython|[c]net2' | grep -v grep | head -10 || echo NO-PYTHON-PROC
echo '--- STAGE1-FORMAL-DIR ---'
ls -lt /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -8 || echo NO-STAGE1-FORMAL-DIR
ls -lt /Users/arm/junyang/a4_selection_20261114/ 2>/dev/null | head -5
echo '--- RECENT-LOGS ---'
ls -lt /Users/arm/junyang/*.log 2>/dev/null | head -8
echo '--- MEM-DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1610 STUDIO-END"
STEOF
echo "$MARK END rc=$?"
} | tee "$LOG"
