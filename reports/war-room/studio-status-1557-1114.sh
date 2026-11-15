#!/bin/bash
# Studio status probe 2026-11-14 15:57: m7bsr1 A4 selection near ETA (15:55-16:00).
# Last read 15:45: driver 42374, 16 workers, ep0-ep7 (8/20), ~43.7s/ep.
# This trip: is selection done? ep count, exit state, selection.json final?
MARK=STUDIOSTAT1557B
echo "$MARK BEGIN $(date '+%F %T %Z')"
~/workspace/studio_ssh.sh "bash -s" <<'STEOF'
echo "STUDIOSTAT1557B STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- A4-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'a4_select|cnet2_train' | grep -v grep || echo NO-A4-PROC
echo '--- A4-LOG ---'
ls -l /Users/arm/junyang/a4_select_m7bsr1_20261114.log 2>/dev/null || echo NO-A4-LOG
echo '--- A4-LOG-TAIL ---'
tail -25 /Users/arm/junyang/a4_select_m7bsr1_20261114.log 2>/dev/null || echo NO-A4-LOG
echo '--- SELECTION-JSON ---'
ls -l /Users/arm/junyang/a4_selection_20261114/m7bsr1_selection.json 2>/dev/null; tail -c 1500 /Users/arm/junyang/a4_selection_20261114/m7bsr1_selection.json 2>/dev/null || echo NO-SELECTION-JSON
echo '--- OTHER-TRAIN-PROC ---'
ps -ax -o pid,%cpu,etime,command | grep -E 'cnet2|train|fable|distill|relabel' | grep -v grep | grep -v a4_select || echo NO-OTHER-TRAIN
echo '--- MEM-DISK ---'
df -h / | tail -1; sysctl -n hw.memsize 2>/dev/null
echo "STUDIOSTAT1557B STUDIO-END"
STEOF
echo "$MARK END rc=$?"
