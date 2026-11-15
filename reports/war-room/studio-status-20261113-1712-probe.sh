#!/bin/bash
# Studio status probe 2026-11-13 17:12 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot started 16:26, run_id 7a189805a8fbc6ec, PID 89571 + 12 workers, 47 scenes du.
# 16:59 trip: PID 89571 etime 32:05, load 12.85, pilot_du.json NOT landed (32 min no output) -> flagged STALLED per 30-min rule, reported to war-room 17:03 sync.
# Owner decision (16:48 widget: wait vs stop-switch to v1.2 formal pilot) still pending as of 17:03 sync.
# This trip: pilot still running? pilot_du.json landed? Any new output?
MARK=STUDIOSTAT1712Q7R3
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-PROC ---'
ps -p 89571 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-89571-NOT-FOUND"
echo "children-of-89571: $(pgrep -P 89571 2>/dev/null | wc -l)"
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'pilot|stage1|stage0_teacher' | grep -v grep || echo NO-PILOT-PROC
echo '--- PILOT-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/ 2>/dev/null | head -20
echo '--- PILOT-LOG ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.json 2>/dev/null
tail -5 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log 2>/dev/null
echo '--- PILOT-JSON-HEAD ---'
head -c 600 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.json 2>/dev/null || echo "NO-pilot_du.json"
echo
echo "$MARK STUDIO-END"
