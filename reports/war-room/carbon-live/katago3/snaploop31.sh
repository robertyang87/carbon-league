#!/bin/bash
# Snapshot loop for KataGo three-leg field: pull all six live files from spot-29 every 120s.
# Self-terminates after end marker seen and all files at 200 lines pushed once more.
OUT=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h31
mkdir -p "$OUT"
LOG=/home/hatch/workspace/carbon-live/katago3/snaploop31.log
END=/root/carbon/katago3_end.txt
TAGS="kgattr_a kgattr_b kgduel_a kgduel_b kgref_a kgref_b"
pull_all() {
  for t in $TAGS; do
    timeout -k 5 60 rsync -e "ssh -F /home/hatch/.ssh/carbon_config" --no-owner --no-group \
      "spot31:/root/carbon/data/bigeval_live/$t.jsonl" "$OUT/" 2>/dev/null
  done
}
counts() {
  local s=""
  for t in $TAGS; do
    local n=0
    [ -f "$OUT/$t.jsonl" ] && n=$(wc -l < "$OUT/$t.jsonl")
    s="$s $t=$n"
  done
  echo "$s"
}
all200() {
  for t in $TAGS; do
    local n=0
    [ -f "$OUT/$t.jsonl" ] && n=$(wc -l < "$OUT/$t.jsonl")
    [ "$n" -ge 200 ] || return 1
  done
  return 0
}
for i in $(seq 1 90); do
  pull_all
  echo "$(date '+%F %T %Z') SNAP$(counts)" >> "$LOG"
  if timeout -k 5 20 ssh -F /home/hatch/.ssh/carbon_config spot31 "test -f $END" 2>/dev/null && all200; then
    echo "$(date '+%F %T %Z') FINAL$(counts)" >> "$LOG"
    break
  fi
  sleep 120
done
echo "$(date '+%F %T %Z') SNAPLOOP_EXIT" >> "$LOG"
