#!/bin/bash
# Studio status probe 2026-11-13 16:42 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot started 16:26, run_id 7a189805a8fbc6ec, PID 89571 + 12 workers, 47 scenes du; 16:29 trip verified three-piece start, pilot_du.json not yet landed.
# This trip: check pilot still running / progress / pilot_du.json landed / relay 88066 state.
MARK=STUDIOSTAT1642K9M2
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
echo '--- RELAY-88066 ---'
ps -p 88066 -o pid,%cpu,etime,command 2>/dev/null || echo "RELAY-88066-GONE"
echo "$MARK STUDIO-END"
