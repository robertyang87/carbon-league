#!/bin/bash
# War-room-sync 10:37 trip (v130): L20 ppou2_m it=20 checkpoint + control-arm resume check. READ-ONLY. Marker L20SYNC1033.
# Context: v129 10:22 probe m it=19 kl 0.1211 kl_coef 0.05, c parked it=4 since 02:41:41;
# it=20 checkpoint est ~10:30, 0.13 single-round KL stop may trigger it=20-22; serial driver
# schedule head ('m',20) then switch to c. Driver log last poll 10:16:50, no polls since.
MARK=L20SYNC1033
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1033-l20probe.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1033 L20-BEGIN $(date '+%F %T %Z')"
echo '--- PPOU2-PROCS ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-260 || echo NO-TRAINER-PROC
echo '--- SPAWN-MAIN-COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- GPU-COMPUTE-APPS ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
echo '--- STATE-LAST ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/state_last.pt /root/carbon/data/rl3/ppou2_c/state_last.pt 2>&1
echo '--- PPOU2_M-CKPTS ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/ 2>&1 | grep -E 'ckpt|state' | tail -12
echo '--- PPOU2_M-TRAIN-TAIL ---'
D=/root/carbon/data/rl3/ppou2_m
wc -l $D/train.jsonl 2>&1; tail -3 $D/train.jsonl 2>&1 | cut -c1-500
echo '--- PPOU2_M-EVAL-TAIL ---'
grep -i eval $D/train.jsonl 2>&1 | tail -3 | cut -c1-500
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/ 2>&1 | grep -iE 'eval' | tail -5
echo '--- PPOU2_M-LOG-TAIL ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m.log 2>&1; tail -4 /root/carbon/data/rl3/ppou2_m.log 2>&1 | cut -c1-300
echo '--- PPOU2_C-TRAIN-TAIL ---'
C=/root/carbon/data/rl3/ppou2_c
wc -l $C/train.jsonl 2>&1; tail -1 $C/train.jsonl 2>&1 | cut -c1-500
ls -l --time-style=+%F_%T $C/state_last.pt 2>&1
echo "L20SYNC1033 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "exit=$?"
cat "$LOG"
