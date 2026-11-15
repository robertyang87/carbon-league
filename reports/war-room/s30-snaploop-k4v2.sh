#!/bin/bash
# C2 2-minute snapshot loop for spot-30 k4v2r1 exams (owner rule 2026-11-12: sync every 2 min).
# Pulls full live files via Tailscale SSH; atomic replace; logs line counts.
HID=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
STORE=$HID/exam_snapshots/h30
mkdir -p "$STORE"
LOG=$STORE/snaploop-k4v2.log
echo "SNAPLOOP-K4V2-START $(date '+%F %T %Z') pid=$$" >> "$LOG"
while true; do
  for tag in k4v2r1_m2 k4v2r1_echo2; do
    tmp="$STORE/.$tag.tmp"
    if timeout -k 10 100 "$HID/s30ssh.sh" "cat /root/carbon/data/bigeval_live/$tag.jsonl" > "$tmp" 2>/dev/null && [ -s "$tmp" ]; then
      mv "$tmp" "$STORE/$tag.jsonl"
      echo "$(date '+%F %T') $tag lines=$(wc -l < "$STORE/$tag.jsonl") bytes=$(stat -c%s "$STORE/$tag.jsonl")" >> "$LOG"
    else
      rm -f "$tmp"
      echo "$(date '+%F %T') $tag PULL-MISS" >> "$LOG"
    fi
  done
  sleep 120
done
