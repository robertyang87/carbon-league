#!/bin/bash
# C2 snapshot loop: pull KataGo remain-600 live files from spot-32 (100.95.45.69)
# to the local h32 mirror every 120 s. Bounded: self-stops after 90 rounds (~3 h)
# or when /root/carbon/data/katago3_remain600_end.txt appears on-machine.
# Verifies by LINE-COUNT GROWTH, never by file presence alone.
# Usage: chmod +x snaploop32.sh; setsid ./snaploop32.sh >> snaploop32.log 2>&1 < /dev/null &
H=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
MIRROR=$H/exam_snapshots/h32
LOG=$H/snaploop32.log
mkdir -p "$MIRROR"
MAX_ROUNDS=90
ROUND=0
while [ "$ROUND" -lt "$MAX_ROUNDS" ]; do
  ROUND=$((ROUND+1))
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  # Stop early if the field's end marker is on-machine.
  if timeout 40 "$H/s32ssh.sh" 'test -f /root/carbon/data/katago3_remain600_end.txt' 2>/dev/null; then
    echo "$TS round=$ROUND END_MARKER_SEEN stopping" >> "$LOG"
    break
  fi
  for tag in kgduel_b kgref_a kgref_b; do
    timeout 60 "$H/s32scp.sh" "root@100.95.45.69:/root/carbon/data/bigeval_live/${tag}.jsonl" "$MIRROR/${tag}.jsonl.tmp" 2>/dev/null \
      && mv "$MIRROR/${tag}.jsonl.tmp" "$MIRROR/${tag}.jsonl"
  done
  COUNTS=""
  for tag in kgduel_b kgref_a kgref_b; do
    if [ -f "$MIRROR/${tag}.jsonl" ]; then
      COUNTS="$COUNTS $tag=$(wc -l < "$MIRROR/${tag}.jsonl")"
    else
      COUNTS="$COUNTS $tag=missing"
    fi
  done
  echo "$TS round=$ROUND$COUNTS" >> "$LOG"
  sleep 120
done
echo "$(date '+%Y-%m-%d %H:%M:%S') snaploop32 DONE rounds=$ROUND" >> "$LOG"
