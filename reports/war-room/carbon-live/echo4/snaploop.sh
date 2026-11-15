#!/bin/bash
# Echo4 exam snapshot loop: pull the three live files from spot-34 every 120s.
DEST=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29
LOG=/home/hatch/workspace/carbon-live/echo4/snaploop.log
mkdir -p "$DEST"
for i in $(seq 1 60); do
  for t in e4pack_m2 e4bare_m2 e4pack_e2; do
    timeout -k 5 50 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
      "spot34:/root/carbon/data/bigeval_live/$t.jsonl" "$DEST/$t.jsonl" 2>/dev/null
  done
  n1=$(wc -l < "$DEST/e4pack_m2.jsonl" 2>/dev/null || echo 0)
  n2=$(wc -l < "$DEST/e4bare_m2.jsonl" 2>/dev/null || echo 0)
  n3=$(wc -l < "$DEST/e4pack_e2.jsonl" 2>/dev/null || echo 0)
  echo "$(date '+%F %T') SNAP$i pack_m2=$n1 bare_m2=$n2 pack_e2=$n3" >> "$LOG"
  END=$(timeout -k 5 30 ssh -n -F /home/hatch/.ssh/carbon_config spot34 "cat /root/carbon/echo4_end.txt 2>/dev/null" 2>/dev/null)
  if [ -n "$END" ] && [ "$n1" = 200 ] && [ "$n2" = 200 ] && [ "$n3" = 200 ]; then
    echo "$(date '+%F %T') FINAL all 200, end=$END" >> "$LOG"
    break
  fi
  sleep 120
done
echo "$(date '+%F %T') SNAPLOOP_EXIT" >> "$LOG"
