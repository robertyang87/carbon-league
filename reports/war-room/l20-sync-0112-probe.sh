#!/bin/bash
# L20 ppou2 main-arm death follow-up probe 2026-11-15 ~01:12 (war-room-sync trip v92).
# Context: 01:10 read showed PID 91223 (main arm m) gone, arm m dir untouched
# since 00:06:28 (train/warmup 0 lines, no state files), spawn_main 15 (was 30),
# control arm c alive with 2 rounds (ckpt_2 01:07:51). Find death evidence:
# launcher log / live dir / failed archive / OOM / monitor processes, plus
# arm c train.jsonl last line (it=2 kl) and GPU PID 91251 identity. READ-ONLY.
MARK=L20SYNC0112
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0112-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0112 L20-BEGIN $(date '+%F %T %Z')"
echo '--- PID-91251 ---'
ps -p 91251 -o pid,ppid,etime,args 2>&1 | cut -c1-220
echo '--- ARM-M-LIVE-DIR ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m/live/ 2>&1
echo '--- ARM-M-LOGS-NEARBY ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_m.failed* /root/carbon/data/rl3/ppou2_m*.log /root/carbon/*.log 2>&1 | head -20
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ | grep -iE "ppou|fail|nohup" 2>&1
echo '--- MONITORS ---'
ps -eo pid,etime,args | grep -iE '[f]irst_round_watch|[a]nchor|[c]heck_arms' | cut -c1-200 || echo NO-MONITOR-PROC
echo '--- ARM-C-TRAIN-LAST ---'
tail -2 /root/carbon/data/rl3/ppou2_c/train.jsonl 2>&1 | cut -c1-400
echo '--- ARM-C-EVAL ---'
cat /root/carbon/data/rl3/ppou2_c/eval.jsonl 2>&1 | cut -c1-400
echo '--- DMSG-OOM-RECENT ---'
dmesg -T 2>/dev/null | grep -iE "oom|killed process" | tail -5 || echo NO-DMESG
echo '--- TRAINER-LOGS-MTIME ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_c/live/ 2>&1 | head
date '+%F %T %Z'
echo "L20SYNC0112 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
