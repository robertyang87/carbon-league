#!/bin/bash
# L20 relaunch health re-probe 2026-11-14 ~01:33 (war-room-sync trip).
# Context: trainer PID 34214 relaunched ~01:27:11 from it79; log tail showed
# one new JSONDecodeError traceback (count 988->989) with log mtime 01:25:51,
# before the trainer start. Determine: (a) what file the JSON error came
# from, (b) whether the trainer is progressing (log/GPU/workers growth).
# READ-ONLY.

MARK=L20SYNC0126B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0126b-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0126B L20-BEGIN $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
echo '--- TRAINER-PROC ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-160 || echo NO-TRAINER-PROC
echo '--- TRAINER-CHILD-COUNT ---'
pgrep -P 34214 | wc -l
echo '--- GPU-PROCS ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1
echo '--- LOG-SIZE-MTIME ---'
wc -l $LG; ls -l $LG; date -r $LG '+%F %T %Z'; date '+%F %T %Z'
echo '--- JSONERR-CONTEXT ---'
JL=$(grep -n 'JSONDecodeError' $LG | tail -1 | cut -d: -f1)
echo "JSONERR-LINE=$JL"
sed -n "$((JL-40)),$((JL+2))p" $LG | grep -vE '^\s+(File "[^"]*", line|\^)' | head -40
echo '--- LOG-TAIL-15 ---'
tail -15 $LG
echo '--- TRAIN-JSONL ---'
wc -l $D/train.jsonl
echo '--- ROLLBACK-STATUS ---'
cat /root/carbon/data/rl3/snipe_n1/state_backup_status.txt 2>/dev/null | tail -3
echo "L20SYNC0126B L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
