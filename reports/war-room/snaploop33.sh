#!/bin/bash
# C2 snapshot loop: pull KataGo kgref_b live file from spot-33 (100.90.180.19)
# to the local h33 mirror every 120 s. Bounded: self-stops after 90 rounds (~3 h)
# or when /root/carbon/katago3_kgrefb_end.txt appears on-machine.
# Verifies by LINE-COUNT GROWTH, never by file presence alone.
# Usage: chmod +x snaploop33.sh; setsid ./snaploop33.sh >> snaploop33.log 2>&1 < /dev/null &
H=~/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files
MIRROR=$H/exam_snapshots/h33
LOG=$H/snaploop33.log
mkdir -p "$MIRROR"
MAX_ROUNDS=90
ROUND=0
while [ "$ROUND" -lt "$MAX_ROUNDS" ]; do
  ROUND=$((ROUND+1))
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  # Stop early if the leg's end marker is on-machine.
  if timeout 40 "$H/s33ssh.sh" 'test -f /root/carbon/katago3_kgrefb_end.txt' 2>/dev/null; then
    echo "$TS round=$ROUND END_MARKER_SEEN stopping" >> "$LOG"
    break
  fi
  for tag in kgsmoke33 kgref_b; do
    timeout 60 "$H/s33scp.sh" "root@100.90.180.19:/root/carbon/data/bigeval_live/${tag}.jsonl" "$MIRROR/${tag}.jsonl.tmp" 2>/dev/null \
      && mv "$MIRROR/${tag}.jsonl.tmp" "$MIRROR/${tag}.jsonl"
  done
  COUNTS=""
  for tag in kgsmoke33 kgref_b; do
    if [ -f "$MIRROR/${tag}.jsonl" ]; then
      COUNTS="$COUNTS $tag=$(wc -l < "$MIRROR/${tag}.jsonl")"
    else
      COUNTS="$COUNTS $tag=missing"
    fi
  done
  echo "$TS round=$ROUND$COUNTS" >> "$LOG"
  sleep 120
done
echo "$(date '+%Y-%m-%d %H:%M:%S') snaploop33 DONE rounds=$ROUND" >> "$LOG"
