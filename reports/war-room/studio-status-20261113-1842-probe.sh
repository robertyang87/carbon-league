#!/bin/bash
# Studio status probe 2026-11-13 18:42 (Tailscale direct; cloud-assistant dead since 04:13, GitHub Actions usage-blocked).
# Context: formal pilot started 18:17, run_id 4a9ab858c7a9b9a3, main PID 18131 + 12 workers,
# 113 scenes / 34-game isolation pool, PREREG formal=true, gate v1.5 13d89867, train_temp 1.0.
# Sidecar monitor PID 18852, first entry 18:19 (elapsed 157s, worker CPU total 1,879.8s, roots_est 1); next ~18:49.
# War-room sync 18:34: pilot in run, sigma-hat est ~22:00; DGX it570 r1 ep4/g1 ep8 @18:28.
# This probe is READ-ONLY: verify pilot + workers + monitor in register, per-worker CPU accumulation
# (slow vs stuck discriminator), progress file state, pilot log mtime, disk/mem.
MARK=STUDIOSTAT1842R7K2
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-1842-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1842R7K2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-PROC ---'
ps -p 18131 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-18131-NOT-FOUND"
echo "children-of-18131: $(pgrep -P 18131 2>/dev/null | wc -l)"
ps -ax -o pid,ppid,%cpu,time,etime,command | grep -E 'stage1_teacher_gate|stage0_teacher|pilot_du_formal' | grep -v grep || echo NO-PILOT-RELATED-PROC
echo '--- MONITOR-PROC ---'
ps -p 18852 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-18852-NOT-FOUND"
echo '--- FORMAL-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -20
echo '--- PROGRESS-FILE ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.progress.json 2>/dev/null || echo "NO-progress.json"
echo '--- PILOT-LOG-TAIL ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.log /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.json 2>/dev/null
tail -5 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.log 2>/dev/null
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py 2>/dev/null
echo "STUDIOSTAT1842R7K2 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
