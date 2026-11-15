#!/bin/bash
# C2 2-minute snapshot loop for spot-31 it320 triple-arm (owner rule 2026-11-12: sync every 2 min).
# Pulls the six live files via Tailscale SSH; atomic replace; logs line counts.
HID=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
STORE=$HID/exam_snapshots/h31/it320
mkdir -p "$STORE"
LOG=$STORE/snaploop-it320.log
echo "SNAPLOOP-IT320-START $(date '+%F %T %Z') pid=$$" >> "$LOG"
while true; do
  for tag in m9xj320r1_m2 m9xj320r1_echo2 m9xj320g1_m2 m9xj320g1_echo2 m9xj320wr1_m2 m9xj320wr1_echo2; do
    tmp="$STORE/.$tag.tmp"
    if timeout -k 10 100 "$HID/s31ssh.sh" "cat /root/carbon/data/bigeval_live/$tag.jsonl" > "$tmp" 2>/dev/null && [ -s "$tmp" ]; then
      mv "$tmp" "$STORE/$tag.jsonl"
      echo "$(date '+%F %T') $tag lines=$(wc -l < "$STORE/$tag.jsonl") bytes=$(stat -c%s "$STORE/$tag.jsonl")" >> "$LOG"
    else
      rm -f "$tmp"
      echo "$(date '+%F %T') $tag PULL-MISS" >> "$LOG"
    fi
  done
  sleep 120
done
