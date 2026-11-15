#!/bin/bash
# L20 OOM-crash follow-up probe 2026-11-14 ~01:19 (war-room-sync trip).
# Context: 01:17:15 direct read found NO rl2_ppo_train proc, last train row
# it79 (00:32), log tail = CUDA OOM in cnet2_model forward, mtime 01:14:15,
# Traceback count 988 (baseline 147 frozen ~it30), GPU 0% / 12087 MiB used.
# MEMORY update notes §19 rolling-copy watchdog deployed 01:15:23 PID 33517
# (after the crash) — verify it is alive. READ-ONLY.

MARK=L20SYNC0119C3D4
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0119-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0119C3D4 L20-BEGIN $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
echo '--- WATCHDOG-33517 ---'
ps -p 33517 -o pid,etime,args 2>&1 || echo WATCHDOG-33517-NOT-FOUND
echo '--- ANY-PYTHON-PROCS ---'
ps -eo pid,etime,rss,args | grep '[p]ython' | head -30 || echo NO-PYTHON-PROCS
echo '--- GPU-PROCS ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1
echo '--- OOM-COUNT ---'
grep -c 'OutOfMemoryError' $LG 2>&1
echo '--- FIRST-OOM-LINE ---'
grep -n 'OutOfMemoryError' $LG 2>&1 | head -3
echo '--- LAST-OOM-LINE ---'
grep -n 'OutOfMemoryError' $LG 2>&1 | tail -3
echo '--- TRACEBACK-COUNT ---'
grep -c Traceback $LG 2>&1
echo '--- LOG-LINES-TOTAL ---'
wc -l $LG
echo '--- CONTEXT-BEFORE-LAST-OOM (50 lines before last OOM, non-traceback) ---'
LASTOOM=$(grep -n 'OutOfMemoryError' $LG | tail -1 | cut -d: -f1)
START=$((LASTOOM-120))
[ $START -lt 1 ] && START=1
sed -n "${START},$((LASTOOM-1))p" $LG | grep -vE '^\s+(File|return|torch|~~|\^)' | tail -30
echo '--- TAIL-AFTER-LAST-OOM ---'
tail -n +$((LASTOOM+1)) $LG | head -20
echo '--- STATE-FILES-ALL ---'
ls -l $D/state_last.pt* $D/*.prev* $D/state_pre_*.pt 2>&1
echo '--- TRAIN-JSONL-WC ---'
wc -l $D/train.jsonl; tail -1 $D/train.jsonl | cut -c1-120
echo '--- LOG-MTIME ---'
ls -l $LG; date -r $LG '+%F %T %Z'; date '+%F %T %Z'
echo "L20SYNC0119C3D4 L20-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
