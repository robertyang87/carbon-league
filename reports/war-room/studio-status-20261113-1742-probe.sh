#!/bin/bash
# Studio status probe 2026-11-13 17:42 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: Stage A pilot started 16:26, run_id 7a189805a8fbc6ec, PID 89571 + 12 workers, 47 scenes du.
# 17:40 trip: PID 89571 etime 01:13:16, 13 subprocesses, load full, pilot_du.log mtime still 16:26:57, pilot_du.json NOT landed (73 min no output) -> stall deepening.
# S1 v1.3 refresh done 17:40 by this line (teacher 77ba3677..., gate bdd3caed...), disk files only, pilot untouched.
# Owner decision (wait vs stop-switch to v1.3 formal pilot) still pending as of 17:40.
# This trip: pilot still running? pilot_du.json landed? Any new process? Disk-file md5s still v1.3?
MARK=STUDIOSTAT1742Q9M4
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
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py 2>/dev/null
echo "$MARK STUDIO-END"
