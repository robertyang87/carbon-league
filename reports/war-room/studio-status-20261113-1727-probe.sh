#!/bin/bash
# Studio status probe 2026-11-13 17:27 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot started 16:26, run_id 7a189805a8fbc6ec, PID 89571 + 12 workers, 47 scenes du.
# 17:14 trip: PID 89571 etime 47:14, load 13.52, pilot_du.log mtime still 16:26:57, pilot_du.json NOT landed (47 min no output) -> stall deepening, already forwarded to war-room by 17:18 sync.
# Owner decision (wait vs stop-switch to v1.3 formal pilot, gate bdd3caed) still pending as of 17:18 war-room sync.
# 17:12 short-lived probe PID 1794 (S1 fix3 acceptance: probe_pilot_r3.py etc.) exited within ~2 min; launcher still unidentified.
# This trip: pilot still running? pilot_du.json landed? 1794 probe output? Any new process?
MARK=STUDIOSTAT1727H3K7
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-PROC ---'
ps -p 89571 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-89571-NOT-FOUND"
echo "children-of-89571: $(pgrep -P 89571 2>/dev/null | wc -l)"
ps -ax -o pid,ppid,%cpu,time,etime,command | grep -E 'stage1_teacher_gate|probe_pilot|p3a_harness|test_s12' | grep -v grep || echo NO-PILOT-RELATED-PROC
echo '--- PILOT-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/ 2>/dev/null | head -20
echo '--- PILOT-LOG ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.json 2>/dev/null
tail -5 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.log 2>/dev/null
echo '--- PILOT-JSON-HEAD ---'
head -c 600 /Users/arm/junyang/carbon/data/stage1_pilot/pilot_du.json 2>/dev/null || echo "NO-pilot_du.json"
echo
echo '--- PROBE-1794-LEFTOVER ---'
ps -p 1794 -o pid,%cpu,etime,command 2>/dev/null || echo "PID-1794-GONE"
ls -lT /tmp/pilot_r3_out.txt /tmp/p3a_harness.py 2>/dev/null
tail -5 /tmp/pilot_r3_out.txt 2>/dev/null
echo "$MARK STUDIO-END"
