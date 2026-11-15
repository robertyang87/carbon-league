#!/bin/bash
# Studio status probe 2026-11-14 ~04:12 (studio-status-report trip). READ-ONLY.
# Direct Tailscale read (owner rule 2026-11-14: Carbon line bypasses GitHub Actions).
# Verifies idle state after 03:57 trip: smoke done, no training, awaiting terminal pilot order.
MARK=STUDIOSTAT0412K7Q4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0412-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0412K7Q4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- SMOKE PID 44472 GONE? ---'
ps -p 44472 -o pid,etime,args 2>/dev/null || echo PID44472-GONE
echo '--- TRAIN/ACCEPT PROCS ---'
ps -eo pid,etime,ppid,args | grep -E '[r]l2_ppo|[s]tage1|[s]pawn_main|[t]est_teacher|[s]1v4accept|[s]elfplay|[g]en1r|[d]u_endgame' | cut -c1-220 || echo NO-MATCHING-PROCS
echo '--- SPAWN_MAIN COUNT ---'
ps -eo args | grep -c '[s]pawn_main' || echo 0
echo '--- SMOKE ARTIFACTS ---'
for f in $(find /Users/arm/junyang/carbon/data /Users/arm/s12test -maxdepth 4 -name 'smoke_du_endgame.json*' 2>/dev/null); do ls -l "$f"; md5 -q "$f"; done
echo '--- NEW LOGS (data/*.log, newest 6) ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -6
echo '--- NEW FILES SINCE 03:58? ---'
find /Users/arm/junyang/carbon/data -maxdepth 2 -newermt '2026-11-14 03:58' -type f 2>/dev/null | head -10 || echo NONE-NEW
echo '--- CODE DIR MD5 (2523235) ---'
cd /Users/arm/junyang/carbon/code 2>/dev/null && md5 -q stage0_teacher.py stage1_teacher_gate.py cnet2_build.py rl2_ppo_train.py rl2_checkpoint.py 2>/dev/null
echo '--- S12TEST GATE ---'
md5 -q /Users/arm/s12test/stage1_teacher_gate.py 2>/dev/null; ls -l /Users/arm/s12test/stage1_teacher_gate.py.OLD-1ef245cb 2>/dev/null || echo NO-OLD-BACKUP
echo '--- DISK/MEM ---'
df -h / | tail -1; vm_stat 2>/dev/null | head -6
echo '--- PILOT LOG UNCHANGED? ---'
ls -l /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v2.log /Users/arm/junyang/carbon/data/stage1_formal/PARTIAL.md 2>/dev/null
echo "STUDIOSTAT0412K7Q4 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
