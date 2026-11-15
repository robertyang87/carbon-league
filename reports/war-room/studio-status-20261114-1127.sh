#!/bin/bash
# Studio status report 11:27 trip: post-B idle confirmation check. READ-ONLY. Marker STUDIOSTAT1127.
MARK=STUDIOSTAT1127
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-1127-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1127 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAINING PROCS (any launcher/gate/worker/train) ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,%cpu,args | grep -E 'stage1_teacher_gate|spawn_mai[n]|cnet2_train|endgame_p1|launcher' | grep -v grep | cut -c1-200 | head -16 || echo NO-TRAIN-PROCS
echo '--- TOP CPU ---'
ps -eo pid,%cpu,etime,args | sort -k2 -rn | head -6 | cut -c1-200
echo '--- PILOT DIR ---'
ls -l /Users/arm/junyang/carbon/data/gen1r_endgame_p1/ | cut -c1-120
echo '--- DONE MARKERS ---'
ls -l /Users/arm/junyang/carbon/data/gen1r_endgame_p1/p1_launch.json /Users/arm/junyang/carbon/data/gen1r_endgame_p1/*.done 2>/dev/null | cut -c1-160; echo RC=$?
echo '--- NEW FILES SINCE 11:14 ---'
find /Users/arm/junyang/carbon/data/gen1r_endgame_p1 -type f -newermt '2026-11-14 11:14' -exec ls -l {} \; | cut -c1-160 || echo NONE-NEW
echo '--- DISK ---'
df -h / | tail -1
echo "STUDIOSTAT1127 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
