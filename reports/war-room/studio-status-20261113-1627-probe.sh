#!/bin/bash
# Studio status probe 2026-11-13 16:27 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot unfrozen 16:26, war-room receipt run_id 7a189805a8fbc6ec claims PID 89571 + 13 workers, load 2.26->7.51, pilot_du.log PREREG mtime 16:26:57.
# This trip independently verifies the three-piece start: process in place + machine load + fresh output mtime.
MARK=STUDIOSTAT1627Q7R3
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
find /Users/arm/junyang/carbon -name 'pilot_du.log' -maxdepth 4 2>/dev/null
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log 2>/dev/null
tail -5 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log 2>/dev/null
echo '--- RELAY-88066 ---'
ps -p 88066 -o pid,etime,command 2>/dev/null || echo "RELAY-88066-GONE"
echo '--- CODE-S1 ---'
cd /Users/arm/junyang/carbon/code && md5 -q stage0_teacher.py cnet2_search.py packs/cnet2_search.py stage1_teacher_gate.py stage0_guided_match.py cnet2_np.py packs/cnet2_np.py
echo "$MARK STUDIO-END"
