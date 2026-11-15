#!/bin/bash
# v130 follow-up: why did ppou2_m stop at it=20? READ-ONLY. Marker L20SYNC1033B.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1033-l20followup.log
{
echo "L20SYNC1033B BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo '--- M-LOG-LAST-30 ---'
tail -30 /root/carbon/data/rl3/ppou2_m.log 2>&1 | cut -c1-250
echo '--- M-LOG-GREP-STOP-ERR ---'
grep -iE 'stop|exit|error|traceback|killed|oom|max|eval|checkpoint' /root/carbon/data/rl3/ppou2_m.log 2>&1 | tail -15 | cut -c1-250
echo '--- TRAINER-MAXIT-ARG ---'
grep -nE 'max.it|max_it|max-round|max_round' /root/carbon/rl2_ppo_train.py 2>&1 | head -10
echo '--- EVAL-JSONL ---'
cat /root/carbon/data/rl3/ppou2_m/eval.jsonl 2>&1 | tail -5 | cut -c1-400
echo '--- M-DIR-FULL ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/ 2>&1 | tail -15
echo '--- ANY-STOP-MARKER ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ 2>&1 | grep -iE 'stop|halt|done|switch' ; ls /root/carbon/data/rl3/*.json 2>&1 | head
echo '--- WHO-STOPPED-CHECK: recent shell history / cron on L20 ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m.log /root/carbon/data/rl3/ppou2_c.log 2>&1
ps -eo pid,etime,args | grep -iE '[w]atch|[s]erial|[p]pou2' | cut -c1-200 || echo NO-WATCHERS
STEOF
echo "L20SYNC1033B END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "exit=$?"
cat "$LOG"
