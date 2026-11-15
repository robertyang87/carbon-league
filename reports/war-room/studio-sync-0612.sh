#!/bin/bash
# Studio status 05:57 trip: q_gate formal run 51972702 checkpoint. READ-ONLY. Marker STUDIOSTAT0612M7Z3.
MARK=STUDIOSTAT0612M7Z3
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/studio-sync-0612-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0612M7Z3 STUDIO-BEGIN $(date '+%F %T %Z')"
echo '--- HOST ---'
hostname; uptime
echo '--- LAUNCHER-GATE ---'
ps -p 72802,72804 -o pid,etime,args 2>/dev/null || echo NO-72802-72804
echo '--- WORKERS ---'
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,args | grep '[s]pawn_main' | awk '{print $1, $2, $3}' | head -25
echo '--- WORKER-CPUTIME-TOTAL ---'
ps -eo time,args | grep '[s]pawn_main' | awk '{n=split($1,t,":"); if(n==2) s=t[1]*60+t[2]; else if(n==3) s=t[1]*3600+t[2]*60+t[3]; else s=0; tot+=s} END {print "WORKER-CPUTIME-TOTAL " tot}'
echo '--- MEM-DISK ---'
vm_stat | head -8
df -h / | tail -1
echo '--- QGATE-FILES ---'
ls -l /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.log /Users/arm/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1 | cut -c1-200
ls -lt /Users/arm/junyang/carbon/data/stageB_q/ 2>/dev/null | head -8
echo "STUDIOSTAT0612M7Z3 STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
