#!/bin/bash
# pull.sh HOST [USER] [INTERVAL_S]  -- every INTERVAL_S (default 300) copy results back to the Mac
# (rl2 runs, eval live files, generated shards). Safe to run in the background; survives box loss.
H=$1; U=${2:-root}; I=${3:-300}; K=~/.ssh/carbon_aliyun_ed25519
ROOT=$(cd "$(dirname "$0")/../.." && pwd); DST=$ROOT/data/cloud; mkdir -p "$DST"
while true; do
  rsync -az --timeout=120 -e "ssh -i $K -o ConnectTimeout=20" $U@$H:carbon/data/rl2 $U@$H:carbon/data/bigeval_live $U@$H:carbon/data/cloud_logs "$DST/" 2>>"$DST/pull.err" \
    && echo "$(date +%T) pulled $(du -sh "$DST" | cut -f1)" >> "$DST/pull.log"
  sleep "$I"
done
