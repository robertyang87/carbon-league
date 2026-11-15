#!/bin/bash
MARK=L20SYNC1204F
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1204-l20follow.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1204F L20-BEGIN $(date '+%F %T %Z')"
ls -l --time-style=long-iso /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_it20_e2*.jsonl 2>&1
wc -l /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_it20_e2*.jsonl 2>&1
tail -1 /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_it20_e2.jsonl 2>&1 | cut -c1-200
ls -l /root/carbon/probe_m2/conf_done.txt 2>&1
ps -eo pid,etime,args | grep '[b]igeval' | wc -l
echo "L20SYNC1204F L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
