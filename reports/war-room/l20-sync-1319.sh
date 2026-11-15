#!/bin/bash
# War-room-sync 13:03 trip: L20 it93 check. READ-ONLY. Marker L20SYNC1319.
MARK=L20SYNC1319
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-1319-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1319 L20-BEGIN $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
echo '--- TRAINER-PROC ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-260 || echo NO-TRAINER-PROC
echo '--- SPAWN-MAIN-COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- GPU-COMPUTE-APPS ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
echo '--- TRAIN-JSONL ---'
wc -l $D/train.jsonl; tail -2 $D/train.jsonl | cut -c1-400
echo '--- STATE-LAST-MTIME ---'
ls -l $D/state_last.pt 2>/dev/null; date '+%F %T %Z'
echo '--- LOG-MTIME ---'
ls -l $LG; date '+%F %T %Z'
echo '--- LOG-TAIL-3 ---'
tail -3 $LG | cut -c1-400
echo '--- TRACEBACK-COUNT ---'
grep -c 'Traceback' $LG
echo "L20SYNC1319 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
