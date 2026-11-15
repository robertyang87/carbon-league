#!/bin/bash
# C2 snapshot loop: ppou2 it20 vs Echo3 diagnostic field on spot-34.
# Pulls live per-game lines to the local mirror every ~120 s. Transfer errors are LOGGED, never swallowed.
cd "$(dirname "$0")"
mkdir -p exam_snapshots/h34
LOG=snaploop_p20.log
HOST=root@100.119.67.98
echo "$(date -u +%FT%TZ) SNAPLOOP_P20_START" >> "$LOG"
while true; do
  for tag in p20smoke p20_a p20_b; do
    if ./s33scp.sh "$HOST:/root/carbon/data/bigeval_live/$tag.jsonl" "exam_snapshots/h34/$tag.jsonl.tmp" 2>>"$LOG"; then
      mv "exam_snapshots/h34/$tag.jsonl.tmp" "exam_snapshots/h34/$tag.jsonl"
    else
      echo "$(date -u +%FT%TZ) PULL_FAIL $tag" >> "$LOG"
    fi
  done
  a=$(wc -l < exam_snapshots/h34/p20_a.jsonl 2>/dev/null || echo 0)
  b=$(wc -l < exam_snapshots/h34/p20_b.jsonl 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) round a=$a b=$b" >> "$LOG"
  sleep 120
done
