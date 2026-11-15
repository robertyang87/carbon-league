#!/bin/bash
# Studio pilot phase-1 acceptance follow-up, READ-ONLY. Marker STUDIOSTAT0727Q9X4B.
MARK=STUDIOSTAT0727Q9X4B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261114-0727b-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0727Q9X4B STUDIO-BEGIN $(date '+%F %T %Z')"
echo '--- PILOT PROCS FULL ARGS + CPU% ---'
ps -eo pid,etime,ppid,%cpu,args | grep -E '[p]1_launch|[s]tage1_teacher_gate|[r]l2_ppo_train|[s]pawn_main' | cut -c1-300
echo '--- P1 LAUNCH JSON ---'
cat /Users/arm/junyang/carbon/data/gen1r_endgame_p1/p1_launch.json 2>/dev/null || echo NO-LAUNCH-JSON
echo '--- SHARD0 MD5 FILE ---'
cat /Users/arm/junyang/carbon/data/gen1r_endgame_p1/gen1r_shard0_md5.txt 2>/dev/null || echo NO-MD5-FILE
echo '--- P1 DIR LISTING ---'
ls -lt /Users/arm/junyang/carbon/data/gen1r_endgame_p1/ 2>/dev/null
echo '--- P1 LAUNCH OUT (tail 15) ---'
tail -15 /Users/arm/junyang/carbon/data/gen1r_endgame_p1/p1_launch.out 2>/dev/null || echo NO-LAUNCH-OUT
echo '--- ENDGAME P1A LOG (tail 20) ---'
tail -20 /Users/arm/junyang/carbon/data/gen1r_endgame_p1/endgame_p1a.log 2>/dev/null || echo NO-P1A-LOG
echo '--- VERIFY INPUT RESULT ---'
grep -iE 'games|positions|mirror|cap2|OK|FAIL|200' /Users/arm/junyang/carbon/data/gen1r_endgame_p1/p1_launch.out 2>/dev/null | tail -10
echo "STUDIOSTAT0727Q9X4B STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
