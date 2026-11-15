#!/bin/bash
# Studio status probe 2026-11-13 16:57 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot started 16:26, run_id 7a189805a8fbc6ec, PID 89571 + 12 workers, 47 scenes du.
# 16:44 trip: PID 89571 etime 17:22, load 13.20, pilot_du.json NOT landed (~18 min no output, under 30-min threshold).
# 16:48 war-room sync: S1 v1.2 (main 4538fddf) landed; in-flight pilot runs v1.1 -> diagnostic-grade only; owner decision pending (wait vs stop-switch).
# This trip: pilot still running? pilot_du.json landed? 30-min no-output threshold hits 16:56:57 (PREREG 16:26:57).
MARK=STUDIOSTAT1657M4Q8
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
echo '--- S1-ACCEPT-93899 ---'
ps -p 93899 -o pid,%cpu,etime,command 2>/dev/null || echo "PID-93899-GONE"
echo "$MARK STUDIO-END"
