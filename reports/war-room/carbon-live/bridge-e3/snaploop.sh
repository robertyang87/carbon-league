#!/bin/bash
# Snapshot loop for Echo3 bridge field: pull both live files from spot-29 every 120s.
# Self-terminates after end marker seen and both files at 200 lines pushed once more.
OUT=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29
mkdir -p "$OUT"
LOG=/home/hatch/workspace/carbon-live/bridge-e3/snaploop.log
END=/root/carbon/bridge_e3_end.txt
for i in $(seq 1 60); do
  timeout -k 5 60 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
    spot29:/root/carbon/data/bigeval_live/echo3bridge_a.jsonl "$OUT/" 2>/dev/null
  timeout -k 5 60 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
    spot29:/root/carbon/data/bigeval_live/echo3bridge_b.jsonl "$OUT/" 2>/dev/null
  A=$(wc -l < "$OUT/echo3bridge_a.jsonl" 2>/dev/null || echo 0)
  B=$(wc -l < "$OUT/echo3bridge_b.jsonl" 2>/dev/null || echo 0)
  echo "$(date '+%F %T %Z') SNAP a=$A b=$B" >> "$LOG"
  if timeout -k 5 20 ssh -F /home/hatch/.ssh/carbon_config spot29 "test -f $END" 2>/dev/null; then
    timeout -k 5 60 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
      spot29:/root/carbon/data/bigeval_live/echo3bridge_a.jsonl "$OUT/" 2>/dev/null
    timeout -k 5 60 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
      spot29:/root/carbon/data/bigeval_live/echo3bridge_b.jsonl "$OUT/" 2>/dev/null
    A=$(wc -l < "$OUT/echo3bridge_a.jsonl" 2>/dev/null || echo 0)
    B=$(wc -l < "$OUT/echo3bridge_b.jsonl" 2>/dev/null || echo 0)
    echo "$(date '+%F %T %Z') FINAL a=$A b=$B" >> "$LOG"
    break
  fi
  sleep 120
done
echo "$(date '+%F %T %Z') SNAPLOOP_EXIT" >> "$LOG"
