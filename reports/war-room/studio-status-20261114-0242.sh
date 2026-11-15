#!/bin/bash
# Studio status probe 2026-11-14 ~02:42 (studio-status-report trip). READ-ONLY.
MARK=STUDIOSTAT0242K7P9
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0242-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0242K7P9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN/ACCEPT PROCS ---'
ps -eo pid,etime,ppid,args | grep -E '[r]l2_ppo|[s]tage1|[s]pawn_main|[t]est_teacher|[s]1v4accept|[s]elfplay|[g]en1r' | cut -c1-220 || echo NO-MATCHING-PROCS
echo '--- SPAWN_MAIN COUNT ---'
pgrep -c spawn_main || echo 0
echo '--- S1V4ACCEPT DIR ---'
ls -la /Users/arm/s1v4accept/ 2>/dev/null | head -20; du -sh /Users/arm/s1v4accept/ 2>/dev/null
echo '--- TW CANON LOG ---'
tail -8 /tmp/tw_canon.log 2>/dev/null || echo NO-TW-LOG; ls -l /tmp/tw_canon.log 2>/dev/null
echo '--- BRUSH MD5 (main code dir) ---'
cd /Users/arm/junyang/carbon/code 2>/dev/null && md5 -q stage0_teacher.py stage1_teacher_gate.py cnet2_build.py rl2_ppo_train.py rl2_checkpoint.py 2>/dev/null; ls -l rl2_checkpoint.py test_review_fixes.py 2>/dev/null || echo MISSING-NEW-FILES
echo '--- DISK/MEM ---'
df -h / | tail -1; vm_stat 2>/dev/null | head -6
echo '--- GEN1R SHARD0 REPLAYS (smoke feed) ---'
ls -ld /Users/arm/junyang/carbon/data/ds_gen1r_shard0 /Users/arm/junyang/carbon/data/gen1r_shard0* 2>/dev/null || echo NO-SHARD0-DIR
find /Users/arm/junyang/carbon/data -maxdepth 2 -name '*gen1r*' 2>/dev/null | head -20
ls /Users/arm/s1v4accept/data/ 2>/dev/null | head; grep -l USAGE_MARK /Users/arm/s1v4accept/data/* 2>/dev/null | head -3
echo '--- SMOKE LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -8
ls -lt /Users/arm/junyang/carbon/data/*smoke* /Users/arm/junyang/carbon/data/*gen1r* 2>/dev/null | head -10
echo '--- PILOT/WEIGHTS UNCHANGED CHECK ---'
ls -l /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log /Users/arm/junyang/carbon/data/stage1_formal/PARTIAL.md 2>/dev/null
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -5
echo "STUDIOSTAT0242K7P9 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
