#!/bin/bash
# War-room-sync 12:04 trip (v136): L20 conf-exam progress + driver/trainer state. READ-ONLY. Marker L20SYNC1220.
MARK=L20SYNC1220
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1220-l20probe.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1220 L20-BEGIN $(date '+%F %T %Z')"
echo '--- CONF-EXAM-FILES ---'
ls -l --time-style=long-iso /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_base_m2_a.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_base_m2_b.jsonl 2>&1
echo '--- CONF-ROWCOUNTS ---'
wc -l /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_base_m2_a.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_base_m2_b.jsonl 2>&1
echo '--- CONF-TAIL-B ---'
tail -1 /root/carbon/probe_m2/data/bigeval_live/ppou2_conf_base_m2_b.jsonl 2>&1 | cut -c1-300
echo '--- EXAM-PROCS ---'
ps -eo pid,etime,args | grep '[b]igeval' | cut -c1-200 || echo NO-EXAM-PROC
echo '--- TRAINER-PROCS ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-200 || echo NO-TRAINER-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- C-ARM-STATE ---'
wc -l /root/carbon/data/rl3/ppou2_c/train.jsonl; tail -1 /root/carbon/data/rl3/ppou2_c/train.jsonl | cut -c1-250
ls -l /root/carbon/data/rl3/ppou2_c/state_last.pt
echo '--- M-ARM-STATE ---'
wc -l /root/carbon/data/rl3/ppou2_m/train.jsonl; tail -1 /root/carbon/data/rl3/ppou2_m/train.jsonl | cut -c1-250
echo '--- SERIAL-DRIVER ---'
ps -eo pid,etime,args | grep -E '[s]erial|[d]river' | grep -i ppo | cut -c1-200 || echo NO-DRIVER-PROC
ls -l /root/carbon/data/rl3/ppou2_m/ckpt_20.npz 2>&1
echo '--- DRIVER-PID-22667 ---'
ps -p 22667 -o pid,etime,args 2>&1 | cut -c1-200 || echo DRIVER-22667-GONE
echo '--- CONF-VERDICT ---'
ls -l --time-style=long-iso /root/carbon/probe_m2/*.json /root/carbon/probe_m2/*conf* 2>&1 | head -10
echo "L20SYNC1220 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
