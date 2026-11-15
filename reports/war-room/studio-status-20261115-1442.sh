#!/bin/bash
MARK=STUDIOSTAT1115-1442
HF=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
LOG=$HF/studio-status-20261115-1442-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT1115-1442 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- LAUNCHER-GATE ---'
ps -eo pid,etime,time,args | grep -E '[q]_launch|[s]tage1_teacher_gate' | cut -c1-160
echo '--- WORKER-COUNT ---'
ps -eo args | grep -c '[s]pawn_main'
echo '--- WORKER-CPU-TOTAL ---'
ps -eo time,args | grep '[s]pawn_main' | awk -F: '{split($1,a,":"); if (NF==3) s+=a[1]*3600+a[2]*60+a[3]; else s+=a[1]*60+a[2]} END {print s}'
echo '--- Q-ARTIFACT ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json /Users/arm/junyang/carbon/data/stageB_q/*.progress.json /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log 2>&1
echo '--- GATE-RESULT-COUNT ---'
ls /Users/arm/junyang/carbon/data/stageB_q/ 2>&1 | grep -i -E 'gate|result|json' | head -20
echo '--- DISK ---'
df -h / /Users/arm/junyang/carbon 2>/dev/null | tail -3
echo "STUDIOSTAT1115-1442 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
