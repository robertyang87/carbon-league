#!/bin/bash
# Snapshot loop for the it=45 exam: rsync the three live files from spot-34
# every ~2 min into the local exam snapshot dir until the exam end marker
# exists AND all files are at full length, then stop.
DEST=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29
LOG=/home/hatch/workspace/carbon-live/it30-exam/snaploop-it45.log
mkdir -p "$DEST"
cnt() { wc -l < "$1" 2>/dev/null || echo 0; }
for i in $(seq 1 90); do
  for t in m45_e4 m45base_e4 m45_e4diag; do
    timeout -k 5 60 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
      "spot34:/root/carbon/data/bigeval_live/$t.jsonl" "$DEST/$t.jsonl" >/dev/null 2>&1
  done
  n1=$(cnt "$DEST/m45_e4.jsonl")
  n2=$(cnt "$DEST/m45base_e4.jsonl")
  n3=$(cnt "$DEST/m45_e4diag.jsonl")
  echo "$(date '+%F %T') SNAP$i m45_e4=$n1 m45base_e4=$n2 m45_e4diag=$n3" >> "$LOG"
  END=$(timeout -k 5 30 ssh -n -F /home/hatch/.ssh/carbon_config spot34 "cat /root/carbon/it45_end.txt 2>/dev/null" 2>/dev/null)
  if [ -n "$END" ] && [ "$n1" = 2000 ] && [ "$n2" = 2000 ] && [ "$n3" = 1000 ]; then
    echo "$(date '+%F %T') FINAL m45_e4=2000 m45base_e4=2000 m45_e4diag=1000, end=$END" >> "$LOG"
    break
  fi
  sleep 120
done
echo "$(date '+%F %T') SNAPLOOP_EXIT" >> "$LOG"
