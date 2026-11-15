#!/bin/bash
# L20 fresh probe 2026-11-14 ~01:27 (war-room-sync trip).
# Context: OOM crash 01:14:15 already forwarded 01:19; relaunch_it79.py
# appeared in ~/workspace/carbon-live at 01:26:57. Check whether a new
# trainer started, whether the 7 orphan spawn_main workers were cleared,
# and whether the §19 watchdog is still alive. READ-ONLY.

MARK=L20SYNC0126X7Q2
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0126-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0126X7Q2 L20-BEGIN $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
echo '--- TRAINER-PROC ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' || echo NO-TRAINER-PROC
echo '--- WATCHDOG-33517 ---'
ps -p 33517 -o pid,etime,args 2>&1 || echo WATCHDOG-33517-NOT-FOUND
echo '--- ORPHAN-SPAWN-MAIN ---'
ps -p 3842,18680,19674,24188,27743,28244,30484 -o pid,etime,args 2>&1 || echo ORPHANS-GONE
echo '--- SPAWN-MAIN-COUNT ---'
pgrep -fc 'spawn_main' 2>&1
echo '--- GPU-PROCS ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
echo '--- GPU-MEM ---'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1
echo '--- RELAUNCH-SCRIPT-ON-L20 ---'
ls -l /root/carbon/relaunch_it79.py /root/carbon/code/relaunch_it79.py 2>&1; md5sum /root/carbon/relaunch_it79.py 2>/dev/null || true
echo '--- TRAIN-JSONL ---'
wc -l $D/train.jsonl; tail -1 $D/train.jsonl | cut -c1-200
echo '--- LOG-MTIME-OOM-TB ---'
ls -l $LG; date -r $LG '+%F %T %Z'; grep -c 'OutOfMemoryError' $LG; grep -c Traceback $LG
echo '--- LOG-TAIL-5 ---'
tail -5 $LG
echo '--- STATE-LAST ---'
ls -l $D/state_last.pt $D/*.prev* 2>&1; cat /root/carbon/data/rl3/snipe_n1/state_backup_status.txt 2>/dev/null | tail -5
echo "L20SYNC0126X7Q2 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
