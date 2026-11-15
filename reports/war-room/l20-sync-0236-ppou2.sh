#!/bin/bash
# War-room-sync 02:36 trip (v98): L20 ppou2_m post-switch liveness check. READ-ONLY. Marker L20SYNC0236.
# Context: serial driver launched m fresh 01:53:55 (PID 99683), verified OK 01:55:32;
# driver log ends 02:01:37 poll error (proxy broken pipe, VM restart window), no retry lines,
# driver process absent -> verify m itself is still alive and progressing.
MARK=L20SYNC0236
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0236-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0236 L20-BEGIN $(date '+%F %T %Z')"
echo '--- PPOU2-PROCS ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-300 || echo NO-TRAINER-PROC
echo '--- SPAWN-MAIN-COUNT ---'
pgrep -f 'spawn_mai[n]' | wc -l
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- GPU-COMPUTE-APPS ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
echo '--- PPOU2_M-DIR ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/ 2>&1 | head -30
echo '--- PPOU2_M-TRAIN-JSONL ---'
D=/root/carbon/data/rl3/ppou2_m
wc -l $D/train.jsonl 2>&1; tail -2 $D/train.jsonl 2>&1 | cut -c1-400
echo '--- PPOU2_M-LOG-MTIME-TAIL ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m.log 2>&1; tail -3 /root/carbon/data/rl3/ppou2_m.log 2>&1 | cut -c1-300
echo '--- PPOU2_C-STATE ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_c/state_last.pt 2>&1
echo "L20SYNC0236 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
