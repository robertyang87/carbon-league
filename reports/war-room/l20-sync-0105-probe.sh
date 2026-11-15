#!/bin/bash
# L20 ppou2 dual-arm health probe 2026-11-15 ~01:05 (war-room-sync trip v92).
# Context: arms relaunched 00:06:26, PIDs 91223/91224, v3 launcher f395bf3c.
# 00:09:05 read passed crash point (GPU 2877MiB/94%, league 11 SNAP keys),
# logs 0 bytes, anchor state md5 + first-round achieved KL still pending.
# Verify now: PIDs alive, GPU occupied, first round landed (train.jsonl /
# warmup), anchor state md5. READ-ONLY.
MARK=L20SYNC0105
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0105-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0105 L20-BEGIN $(date '+%F %T %Z')"
echo '--- ARM-PROCS ---'
ps -eo pid,etime,args | grep '[l]aunch_ppou2' | cut -c1-200 || echo NO-LAUNCHER-PROC
ps -p 91223,91224 -o pid,etime,args 2>&1 | cut -c1-200
echo '--- SPAWN-MAIN-COUNT ---'
ps -eo pid,etime,args | grep '[s]pawn_main' | wc -l
echo '--- GPU ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- CHECK-ARMS ---'
python3 /root/carbon/check_arms.py 2>&1
echo '--- ARM-DIRS ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/ 2>&1
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_c/ 2>&1
echo '--- ANCHOR-STATE-MD5 ---'
for f in /root/carbon/data/rl3/ppou2_m/state*.pt /root/carbon/data/rl3/ppou2_c/state*.pt; do
  [ -e "$f" ] && md5sum "$f"
done
echo '--- LOG-MTIME ---'
date '+%F %T %Z'
echo "L20SYNC0105 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
