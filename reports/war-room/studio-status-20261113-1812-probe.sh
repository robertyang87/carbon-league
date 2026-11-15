#!/bin/bash
# Studio status probe 2026-11-13 18:12 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: 17:59 trip: PID 89571 etime 01:32:13, 11/12 workers STOPPED, only 89578 at ~99%;
# no output 92 min 13 s, pilot_du.json NOT landed, pilot_du.log PREREG-only 1,597 B (mtime 16:26:57).
# S1 disk files v1.3 (teacher 77ba3677..., gate bdd3caed...). Disk 421Gi free.
# NEW since last trip (war-room 18:10, owner-authorized strategy-group + war-room joint decision, §12 v1.5 HEAD 002b262b):
# (b) gate extract_scene now observes along replay prefix -> gate 13d89867 in main; (c) deferred;
# v1.1 pilot STOPPED (sole executor Studio); formal pilot only after gate refresh + worker-stop-cause conclusion,
# on DGX 34-game isolation pool. This probe is READ-ONLY: verify whether the stop has been executed,
# whether 89578 also stopped, whether pilot_du.json landed, and whether gate on disk is still v1.3 or refreshed to v1.5.
MARK=STUDIOSTAT1812K7P3
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-1812-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1812K7P3 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-PROC ---'
ps -p 89571 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-89571-NOT-FOUND"
echo "children-of-89571: $(pgrep -P 89571 2>/dev/null | wc -l)"
ps -ax -o pid,ppid,%cpu,time,etime,command | grep -E 'stage1_teacher_gate|probe_pilot|p3a_harness|test_s12|stage0_teacher' | grep -v grep || echo NO-PILOT-RELATED-PROC
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
echo "STUDIOSTAT1812K7P3 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
