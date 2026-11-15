#!/bin/bash
# War-room-sync 11:52 trip (v134): L20 ppou2_m it=20 checkpoint exam COMPLETION VERIFY + relaunch state. READ-ONLY. Marker L20SYNC1152.
# Context: memory 11:16-11:20: E2 field DONE_MARKER 11:16:06, both fields 2000/2000 err 0, merged dU +0.0466 CI [0.0367,0.0568]
# (M2 +0.0399, E2 +0.0534); strategy-group verdict survival line PASSED; training relaunched kl 0.15, control arm running
# (pid 221998, verify OK 11:18:48 per L20 line). This trip: INDEPENDENT verify of live files + control-arm three-piece + driver/watchdog.
MARK=L20SYNC1152
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1152-l20probe.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1152 L20-BEGIN $(date '+%F %T %Z')"
echo '--- EXAM-FILES ---'
ls -l --time-style=+%F_%T /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl 2>&1
echo '--- ROWCOUNTS ---'
wc -l /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl 2>&1
echo '--- E2-TAIL ---'
tail -1 /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl | cut -c1-400
echo '--- DONE-MARKER ---'
ls -l --time-style=+%F_%T /root/carbon/probe_m2/it20_done.txt /root/carbon/data/rl3/it20_done.txt 2>&1
cat /root/carbon/probe_m2/it20_done.txt 2>/dev/null | cut -c1-300
echo '--- RECENT-EXAM-FILES ---'
ls -lt --time-style=+%F_%T /root/carbon/probe_m2/data/bigeval_live/ 2>/dev/null | head -8
echo '--- EXAM-PROCS ---'
ps -eo pid,etime,args | grep '[b]igeval.py exp/ppou2_m_it20' | cut -c1-200 || echo NO-EXAM-PROC
echo '--- TRAINER-PROCS ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-200 || echo NO-TRAINER-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- C-ARM-STATE ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_c/state_last.pt 2>&1
wc -l /root/carbon/data/rl3/ppou2_c/train.jsonl; tail -1 /root/carbon/data/rl3/ppou2_c/train.jsonl | cut -c1-300
echo '--- M-ARM-STATE ---'
wc -l /root/carbon/data/rl3/ppou2_m/train.jsonl; tail -1 /root/carbon/data/rl3/ppou2_m/train.jsonl | cut -c1-300
echo '--- SERIAL-DRIVER ---'
ps -eo pid,etime,args | grep '[s]erial_driver' | cut -c1-200 || echo NO-DRIVER-PROC
echo "L20SYNC1152 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "exit=$?"
cat "$LOG"
