#!/bin/bash
# War-room-sync 11:05 trip (v132): L20 ppou2_m it=20 checkpoint exam FIRST RATE READ. READ-ONLY. Marker L20SYNC1105.
# Context: v130 10:44 forwarded it=20 checkpoint landed 10:27:22 (kl 0.1288), trainer stopped ~10:28,
# checkpoint exam opened ~10:32 (Master2 field seeds 980000-981999, 2000 games, tag ppou2_m_it20_m2, 24 workers);
# 10:34:34 live jsonl already 230 rows. This trip: live row count + rate + ETA. Echo2 field follows.
MARK=L20SYNC1105
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1105-l20probe.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC1105 L20-BEGIN $(date '+%F %T %Z')"
echo '--- EXAM-PROCS ---'
ps -eo pid,etime,args | grep '[b]igeval.py exp/ppou2_m_it20' | cut -c1-200 | head -30
echo '--- EXAM-PROC-COUNT ---'
ps -eo args | grep -c '[b]igeval.py exp/ppou2_m_it20'
echo '--- TRAINER-PROCS (expect none) ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' | cut -c1-200 || echo NO-TRAINER-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1
echo '--- EXAM-FILES ---'
find /root/carbon -name 'ppou2_m_it20*' -mmin -180 2>/dev/null | head -20
ls -l --time-style=+%F_%T /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl 2>&1
echo '--- M2-FIELD-LIVE ---'
for f in /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_m2.jsonl /root/carbon/probe_m2/data/bigeval_live/ppou2_m_it20_e2.jsonl; do
  if [ -f "$f" ]; then echo "FILE $f"; wc -l "$f"; tail -1 "$f" | cut -c1-300; fi
done
echo '--- E2-FIELD-ANY ---'
find /root/carbon -name 'ppou2_m_it20_e2*' -mmin -180 2>/dev/null | head -5
ls -l --time-style=+%F_%T /root/carbon/data/rl3/it20_done.txt /root/carbon/probe_m2/it20_done.txt 2>&1
echo '--- M-TRAIN-UNCHANGED-CHECK ---'
wc -l /root/carbon/data/rl3/ppou2_m/train.jsonl; tail -1 /root/carbon/data/rl3/ppou2_m/train.jsonl | cut -c1-200
echo '--- C-ARM-UNCHANGED-CHECK ---'
ls -l --time-style=+%F_%T /root/carbon/data/rl3/ppou2_c/state_last.pt; wc -l /root/carbon/data/rl3/ppou2_c/train.jsonl
echo "L20SYNC1105 L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "exit=$?"
cat "$LOG"
