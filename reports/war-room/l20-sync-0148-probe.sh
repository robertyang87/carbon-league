#!/bin/bash
# L20 relaunch mid-it80 health probe 2026-11-14 ~01:50 (war-room-sync trip).
# Context: trainer PID 34214 relaunched ~01:27:11 from it79 (after 01:14 OOM);
# it80 due ~02:10-02:14 est. Verify mid-round health only: PID alive,
# GPU occupied, log mtime/size advancing, train.jsonl still 79 rows,
# no new JSONDecodeError since relaunch, watchdog alive, no orphan regrowth.
# READ-ONLY.

MARK=L20SYNC0148
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0148-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0148 L20-BEGIN $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
echo '--- TRAINER-PROC ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-200 || echo NO-TRAINER-PROC
echo '--- SPAWN-MAIN-COUNT ---'
ps -eo pid,etime,args | grep '[s]pawn_main' | wc -l
ps -eo pid,etime,args | grep '[s]pawn_main' | cut -c1-120
echo '--- GPU ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- LOG-SIZE-MTIME ---'
wc -l $LG; ls -l $LG; date -r $LG '+%F %T %Z'; date '+%F %T %Z'
echo '--- JSONERR-COUNT ---'
grep -c 'JSONDecodeError' $LG
echo '--- TRACEBACK-COUNT ---'
grep -c 'Traceback' $LG
echo '--- LOG-TAIL-12 ---'
tail -12 $LG
echo '--- TRAIN-JSONL ---'
wc -l $D/train.jsonl; tail -1 $D/train.jsonl | cut -c1-300
echo '--- ROLLBACK-STATUS ---'
ps -eo pid,etime,args | grep '[s]tate_rollbackup' | cut -c1-160 || echo NO-WATCHDOG
cat /root/carbon/data/rl3/snipe_n1/state_backup_status.txt 2>/dev/null | tail -3
ls -l $D/state_last.pt* 2>/dev/null | head -8
echo "L20SYNC0148 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
