#!/bin/bash
# C2 it95 exam snapshot loop (local): pull live per-game lines off spot-31 every 2 min.
# Stops when STOP file appears in the snapshot dir.
DIR=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h31/it95
mkdir -p "$DIR"
pull() {
  timeout -k 10 60 scp -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
    -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p" \
    -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
    root@100.98.67.20:/root/carbon/data/bigeval_live/$1.jsonl "$DIR/$1.jsonl" 2>/dev/null
}
cnt() { if [ -f "$DIR/$1.jsonl" ]; then wc -l < "$DIR/$1.jsonl"; else echo NA; fi }
while true; do
  for f in it95_smoke it95_echo2 m3s2r1_echo2_base95; do pull $f; done
  echo "$(date '+%F %T') it95_smoke=$(cnt it95_smoke) it95_echo2=$(cnt it95_echo2) m3s2r1_echo2_base95=$(cnt m3s2r1_echo2_base95)" >> "$DIR/snaploop.log"
  if [ -f "$DIR/STOP" ]; then echo "$(date '+%F %T') STOP seen, exiting" >> "$DIR/snaploop.log"; break; fi
  sleep 120
done
