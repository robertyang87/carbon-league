#!/bin/bash
# pull.sh HOST [USER] [INTERVAL_S]  -- every INTERVAL_S (default 300) copy results back to the Mac
# (rl2 runs, eval live files, generated shards). Safe to run in the background; survives box loss.
H=$1; U=${2:-root}; I=${3:-300}; K=~/.ssh/carbon_aliyun_ed25519
ROOT=$(cd "$(dirname "$0")/../.." && pwd); DST=$ROOT/data/cloud; mkdir -p "$DST"
while true; do
  # results straight into data/bigeval_live (what paired.py reads); DAgger shards into data/<dir>; logs into data/cloud
  rsync -az --timeout=120 -e "ssh -i $K -o ConnectTimeout=20" $U@$H:carbon/data/bigeval_live/ "$ROOT/data/bigeval_live/" 2>>"$DST/pull.err"
  for d in $(ssh -i $K -o ConnectTimeout=20 $U@$H 'cd carbon/data; ls -d dagger4* 2>/dev/null' </dev/null); do
    rsync -az --timeout=300 -e "ssh -i $K -o ConnectTimeout=20" $U@$H:carbon/data/$d/ "$ROOT/data/$d/" 2>>"$DST/pull.err"; done
  rsync -az --timeout=120 -e "ssh -i $K -o ConnectTimeout=20" --include='*.progress' --include='*.log' --include='*.out' --include='*.done' --exclude='*/' --exclude='*' \
    $U@$H:carbon/data/ "$DST/logs/" 2>>"$DST/pull.err" \
    && echo "$(date +%T) pulled live $(ls "$ROOT/data/bigeval_live" | wc -l) files" >> "$DST/pull.log"
  sleep "$I"
done
