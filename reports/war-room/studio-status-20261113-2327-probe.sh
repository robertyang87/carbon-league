#!/bin/bash
# Studio status probe 2026-11-13 23:27 (Tailscale direct; same pattern as 2312 trip).
# Context since last read 23:14:49: war-room dispatch LANDED — brush to a4dd8329 done
# (8 files md5-verified, backup code_backup_pre_v2_a4dd8329_20261113) and v2 pilot
# re-run started 23:17 (run 49b518f67267742c, main PID 91954 + 12 workers, 113 roots /
# 34 games, --belief sample, teacher v2, gate dd921e57; PREREG all pass).
# Strategy group 23:21 ruled: do NOT restart, run 995300 same roots to completion.
# §17 (HEAD fa7b41c6): after v2 pilot, next job is TERMINAL pilot (gen-1 shard0 step>=200
# roots, v2+sample, ~400 roots, ~11h est) — NOT formal q gate; awaits shard0 landing +
# strategy-group confirmation via war-room. This trip: verify pilot still running with
# the three-piece check (process + load + fresh log/file mtime). READ-ONLY.

MARK=STUDIOSTAT2327K9R4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261113-2327-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT2327K9R4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -4
echo '--- PILOT-MAIN-91954 ---'
ps -p 91954 -o pid,%cpu,%mem,etime,command 2>/dev/null || echo "PID-91954-NOT-FOUND"
echo '--- ANY-TRAINING-PROC ---'
ps aux | grep -iE 'python.*(stage1|stage0|selfplay|rl2_ppo|teacher|train)' | grep -v grep || echo "NO-TRAINING-PROC"
echo '--- TOP-CPU ---'
ps aux -r | head -8
echo '--- S1-FILES-MD5 ---'
md5 /Users/arm/junyang/carbon/code/stage0_teacher.py /Users/arm/junyang/carbon/code/stage1_teacher_gate.py /Users/arm/junyang/carbon/code/cnet2_build.py /Users/arm/junyang/carbon/code/producer/selfplay_gen.py /Users/arm/junyang/carbon/code/rl2_ppo_train.py /Users/arm/junyang/carbon/code/rl2_ppo_cnet2.py /Users/arm/junyang/carbon/code/stage0_guided_match.py /Users/arm/junyang/carbon/code/stage0_measure_teacher.py 2>/dev/null
echo '--- FORMAL-DIR ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/ 2>/dev/null | head -25
echo '--- V2-PILOT-SIGNS ---'
ls -lT /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2* 2>/dev/null || echo "NO-v2-pilot-files"
ls -ld /Users/arm/junyang/carbon/code_backup_pre_v2_a4dd8329_20261113 2>/dev/null || echo "NO-v2-backup-dir"
echo "STUDIOSTAT2327K9R4 STUDIO-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
