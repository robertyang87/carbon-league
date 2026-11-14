#!/bin/bash
# Poll L20 for state_last.pt in both ppou2 out dirs; on first sighting per arm,
# snapshot it to state_anchor_frozen.pt on L20 and record md5 + train line count.
SSH="ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p root@100.79.251.87"
OUT=~/workspace/carbon-live/ppou2/anchor_watch_result.txt
: > "$OUT"
for i in $(seq 1 160); do
  R=$(timeout -k 10 40 $SSH 'bash -c "
for a in m c; do
  d=/root/carbon/data/rl3/ppou2_\$a
  if [ -f \$d/state_last.pt ] && [ ! -f \$d/state_anchor_frozen.pt ]; then
    cp \$d/state_last.pt \$d/state_anchor_frozen.pt
    echo CAPTURED \$a \$(md5sum \$d/state_anchor_frozen.pt | cut -d\" \" -f1) trainlines=\$(wc -l < \$d/train.jsonl) warmlines=\$(wc -l < \$d/warmup.jsonl) \$(date +%T)
  fi
  if [ -f \$d/state_anchor_frozen.pt ]; then echo DONE \$a; fi
done"' 2>/dev/null)
  echo "poll $i $(date +%T): $R" >> "$OUT"
  if [ "$(echo "$R" | grep -c DONE)" = "2" ]; then echo "BOTH_ANCHORS_DONE" >> "$OUT"; break; fi
  sleep 30
done
cat "$OUT"
