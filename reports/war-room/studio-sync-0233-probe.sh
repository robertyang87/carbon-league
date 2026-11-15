#!/bin/bash
# Studio s1v4accept progress probe 2026-11-14 ~02:33 (war-room-sync trip). READ-ONLY.
MARK=STUDIOSYNC0233
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-sync-0233-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSYNC0233 STUDIO-BEGIN $(date '+%F %T %Z')"
echo '--- S1V4ACCEPT PROCS ---'
ps -eo pid,etime,args | grep -E '[s]1v4accept|[t]est_' | grep -v grep | cut -c1-200 || echo NO-ACCEPT-PROCS
echo '--- S1V4ACCEPT DIR ---'
ls -la /Users/arm/s1v4accept/ 2>/dev/null | head -20
echo '--- TW CANON LOG ---'
tail -5 /tmp/tw_canon.log 2>/dev/null || echo NO-TW-LOG
echo '--- ACCEPT LOGS ---'
ls -lt /Users/arm/s1v4accept/*.log /Users/arm/s1v4accept/**/*.log 2>/dev/null | head -10
echo '--- LOAD ---'
uptime
echo "STUDIOSYNC0233 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
