#!/bin/bash
# Studio status probe 2026-11-13 21:42 (Tailscale direct; same pattern as 2127 trip).
# Context: formal pilot run_id 4a9ab858c7a9b9a3, main PID 18131 + 12 workers,
# 113 roots, PREREG formal=true, gate v1.5 13d89867 on disk.
# 2127 trip (read 21:29): pilot in TAIL PHASE, 7th sidecar 21:19:49,
# roots_est 106/113, 5 workers RN / 7 SN, load falling, ETA revised
# ~21:50-22:10 (estimate). Sidecar #8 due ~21:49:48.
# This trip: check whether 8th sidecar landed, whether
# pilot_du_formal.json exists (completion), worker count, load, gate md5.
# War-room latest (§15, 21:18): pilot done -> NO q-gate; brush gate to ac307c9d
# after landing + md5 receipt; then teacher-v2 endgame validation support.
# Also noted: acceptance-preview (Claude diff smoke tests on Studio) is a
# future to-do, not yet due. This probe is READ-ONLY.
MARK=STUDIOSTAT2142Q7R4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2142-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2142Q7R4 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo "STUDIOSTAT2142Q7R4 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
