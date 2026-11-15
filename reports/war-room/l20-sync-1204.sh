#!/bin/bash
# War-room-sync 12:04 trip: L20 it92 landing check. READ-ONLY. Marker L20SYNC1204.
MARK=L20SYNC1204
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-1204-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1204 L20-BEGIN $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
echo '--- TRAINER-PROC ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-260 || echo NO-TRAINER-PROC
echo '--- SPAWN-MAIN-COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- TRAIN-JSONL ---'
wc -l $D/train.jsonl; tail -3 $D/train.jsonl | cut -c1-400
echo '--- STATE-LAST-MTIME ---'
ls -l $D/state_last.pt 2>/dev/null; date '+%F %T %Z'
echo '--- KL010-WATCHDOG ---'
ps -eo pid,etime,args | grep '[k]l010_swap' | cut -c1-160 || echo NO-WATCHDOG
ls -l $D/state_pre_kl010_it92.pt $D/kl010_swap_status.txt 2>/dev/null
echo '--- LOG-TAIL-6 ---'
tail -6 $LG
echo '--- TRACEBACK-COUNT ---'
grep -c 'Traceback' $LG
echo "L20SYNC1204 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
