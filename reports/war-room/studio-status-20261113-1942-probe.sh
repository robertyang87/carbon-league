#!/bin/bash
# Studio status probe 2026-11-13 19:42 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: formal pilot started 18:17, run_id 4a9ab858c7a9b9a3, main PID 18131 + 12 workers,
# 113 scenes / 34-game isolation pool, PREREG formal=true, gate v1.5 13d89867, train_temp 1.0.
# Sidecar monitor PID 18852, 2nd entry 18:49:48 (elapsed 1957s, worker CPU total 23,477.7s, roots_est 19);
# 3rd sidecar landed 19:19:48 (worker CPU total 45,075.1s, roots_est 37); 4th sidecar due ~19:49. ETA ~21:27-21:40 (estimate, ~100s/root).
# Last read 19:29: pilot healthy, load 12.30, 12/12 workers 99-100%, CPU 72:03 vs etime 72:05, 3rd sidecar 19:19:48 landed (worker CPU total 45,075.1s, roots_est 37). 4th sidecar due ~19:49. ETA ~21:27-21:40 (estimate).
# War-room latest: snap_70 calibration GO (Master2 +280, Echo2 +237), GO flag landed L20 19:41, it73 pool-change ~20:04 then cut Terminal-U. No Studio order pending.
# This probe is READ-ONLY: verify pilot + workers + monitor in register, per-worker CPU accumulation
# (slow vs stuck discriminator), progress file state (expect 3rd sidecar ~19:19 landed),
# pilot log mtime, disk/mem.
MARK=STUDIOSTAT1942Q7M4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-1942-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1942Q7M4 STUDIO-BEGIN $(date '+%F %T %Z')"
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
echo "STUDIOSTAT1942Q7M4 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
