#!/bin/bash
# War-room-sync 14:03 trip: Studio v3 pilot 14:03 CPU checkpoint. READ-ONLY. Marker STUDIOSTAT1418.
MARK=STUDIOSTAT1418
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-sync-1418-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1418 STUDIO-BEGIN $(date '+%F %T %Z')"
echo '--- LAUNCHER-GATE ---'
ps -p 89589,89592 -o pid,etime,args 2>/dev/null || echo NO-89589-89592
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,args | grep '[s]pawn_main' | awk '{print $1, $2, $3}' | head -15
echo '--- LOAD ---'
uptime
echo '--- V3-LOG ---'
ls -l /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.log /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.json 2>&1 | cut -c1-160
head -2 /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.log 2>/dev/null | cut -c1-300
echo '--- V3-DIR-NEW ---'
find /Users/arm/junyang/carbon/data/stage1_formal -newer /Users/arm/junyang/carbon/data/stage1_formal/pilot_du_v3.log -type f 2>/dev/null | head -10
echo "STUDIOSTAT1418 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
