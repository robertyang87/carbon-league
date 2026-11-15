#!/bin/bash
# War-room-sync 12:20 trip (v137): spot-34 Echo4 exam in-flight check. READ-ONLY. Marker C1SYNC1220.
MARK=C1SYNC1220
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1220-c1spot34.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.119.67.98 'bash -s' <<'STEOF'
echo "C1SYNC1220 SPOT34-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- ECHO4-EXAM-PROCS ---'
ps -eo pid,etime,args | grep '[b]igeval' | cut -c1-200 || echo NO-BIGEVAL-PROC
echo '--- ECHO4-LIVE-FILES ---'
ls -l --time-style=long-iso /root/carbon/data/bigeval_live/ 2>&1 | grep -i echo4 || echo NO-ECHO4-FILES
wc -l /root/carbon/data/bigeval_live/*echo4* 2>&1 | tail -5
echo '--- SNAPSHOT ---'
ls -l --time-style=long-iso /root/carbon/data/bigeval_live/ 2>&1 | tail -5
echo "C1SYNC1220 SPOT34-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
