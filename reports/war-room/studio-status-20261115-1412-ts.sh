#!/bin/bash
MARK=STUDIOSTAT1115-1412
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-status-20261115-1412-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1115-1412 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname
uptime
echo '--- GATE-PS ---'
ps -eo pid,etime,time,args | grep -E '[q]_launch|[s]tage1_teacher_gate' | cut -c1-160
echo '--- WORKER-COUNT ---'
ps -eo args | grep -c '[s]pawn_main'
echo '--- WORKER-CPU-TOTAL ---'
ps -eo time,args | grep '[s]pawn_main' | awk -F: '{split($1,a,":"); if (NF==3) s+=a[1]*3600+a[2]*60+a[3]; else s+=a[1]*60+a[2]} END {print s}'
echo '--- Q-OUT ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json /Users/arm/junyang/carbon/data/stageB_q/*.progress.json 2>&1
stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S' /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>&1
echo '--- Q-LOG-TAIL ---'
tail -5 /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>&1
echo "STUDIOSTAT1115-1412 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
