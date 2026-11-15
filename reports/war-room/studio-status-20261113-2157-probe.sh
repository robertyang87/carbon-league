#!/bin/bash
# Studio status probe 2026-11-13 21:57 (Tailscale direct; same pattern as 2142 trip).
# Context: formal pilot DONE 21:38:01 (grade formal, run_id 4a9ab858c7a9b9a3).
# Gate v1.6 ac307c9d brushed 21:48 (Studio-line receipt, war-room 21:57 sync confirms).
# Monitor PID 18852 expected to self-stop at ~21:49:49 wake (writes pilot_exited final sidecar).
# War-room latest: strategy group N-lock (cap 492, delta 0.02, tentative); Studio idle
# awaiting teacher v2, NO new dispatch before v2 lands.
# This trip: verify still idle (normal), gate on disk now v1.6, monitor exited or not,
# no new training process, load/disk. READ-ONLY.
MARK=STUDIOSTAT2157M4P9
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2157-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2157M4P9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- ANY-TRAINING-PROC ---'
ps aux | grep -iE 'python.*(stage1|stage0|selfplay|rl2_ppo|teacher|train)' | grep -v grep || echo "NO-TRAINING-PROC"
echo '--- PILOT-PROC ---'
ps -p 18131 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-18131-NOT-FOUND"
echo "children-of-18131: $(pgrep -P 18131 2>/dev/null | wc -l)"
echo '--- MONITOR-PROC ---'
ps -p 18852 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-18852-NOT-FOUND"
echo '--- FORMAL-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -20
echo '--- PROGRESS-FILE ---'
cat /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.progress.json 2>/dev/null || echo "NO-progress.json"
echo '--- PILOT-JSON ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_formal.json* 2>/dev/null
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py 2>/dev/null
ls -ld /Users/arm/junyang/carbon/code_backup_pre_s1v16_20261113 2>/dev/null || echo "NO-v16-backup-dir"
echo "STUDIOSTAT2157M4P9 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
