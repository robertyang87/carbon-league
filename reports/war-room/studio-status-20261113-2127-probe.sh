#!/bin/bash
# Studio status probe 2026-11-13 21:27 (Tailscale direct; same pattern as 2112 trip).
# Context: formal pilot run_id 4a9ab858c7a9b9a3, main PID 18131 + 12 workers,
# 113 roots, PREREG formal=true, gate v1.5 13d89867.
# 2112 trip (21:14:56): pilot alive, load ~9.9-12.4, 6th sidecar 20:49:49,
# roots_est 90/113, out_json not yet present. 7th sidecar due ~21:19:48.
# ETA window 21:27-21:40 opens this trip — check whether 7th sidecar landed
# and whether pilot_du_formal.json exists (completion).
# War-room latest (§15, 21:18): pilot done -> NO q-gate; brush gate to ac307c9d
# after landing + md5 receipt; then teacher-v2 endgame validation support.
# This probe is READ-ONLY.
MARK=STUDIOSTAT2127K9M2
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2127-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2127K9M2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-PROC ---'
ps -p 18131 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-18131-NOT-FOUND"
echo "children-of-18131: $(pgrep -P 18131 2>/dev/null | wc -l)"
echo '--- WORKER-CPU-TIMES ---'
for p in $(pgrep -P 18131 2>/dev/null); do ps -p $p -o pid,stat,%cpu,time,etime,command 2>/dev/null | tail -1; done
echo '--- MONITOR-PROC ---'
ps -p 18852 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-18852-NOT-FOUND"
echo '--- FORMAL-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -20
echo '--- PROGRESS-FILE ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.progress.json 2>/dev/null || echo "NO-progress.json"
echo '--- PILOT-LOG-TAIL ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.log /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.json 2>/dev/null
tail -3 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.log 2>/dev/null | cut -c1-200
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py 2>/dev/null
echo "STUDIOSTAT2127K9M2 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
