#!/bin/bash
# sync_hosts.sh CLOUD_IP [INTERVAL_S] -- keeps the cloud host and the DGX from running the same exam twice.
# Every INTERVAL_S: a result file that is COMPLETE on one host (N games, N from the tag's queue line) is copied
# to the other host only if the other host has NOT started that tag (no file there). The other host's resumable
# queue then skips it. A file that already exists on the destination is never touched (it may be mid-write).
# Complete files also land in data/bigeval_live on the Mac (what paired.py reads).
C=$1; I=${2:-180}; K=~/.ssh/carbon_aliyun_ed25519
ROOT=$(cd "$(dirname "$0")/../.." && pwd); L=$ROOT/data/dgx_live; mkdir -p "$L/sync"
want() {  # expected game count for a tag, from the queue scripts (100 or 200)
  grep -hE " $1\"?$" "$ROOT/code/cloud/q6.sh" "$ROOT/code/cloud/dq1.sh" "$ROOT/code/cloud/q7.sh" 2>/dev/null | awk '{print $(NF-1)}' | tr -d '"' | head -1; }
full() { [ -f "$1" ] && [ "$(grep -c '"margin": -\?[0-9]' "$1")" -ge "$2" ]; }
while true; do
  rsync -az --timeout=120 -e "ssh -i $K" "root@${C}:carbon/data/bigeval_live/" "$L/sync/cloud/" 2>/dev/null
  rsync -az --timeout=120 "dgx-via-spot3:junyang/carbon/data/bigeval_live/" "$L/sync/dgx/" 2>/dev/null
  for src in cloud dgx; do
    dst=$([ $src = cloud ] && echo dgx || echo cloud)
    for f in "$L/sync/$src"/c_*.jsonl; do
      [ -f "$f" ] || continue; t=$(basename "$f" .jsonl); n=$(want "$t"); [ -n "$n" ] || continue
      full "$f" "$n" || continue
      cp -n "$f" "$ROOT/data/bigeval_live/$t.jsonl" 2>/dev/null
      [ -f "$L/sync/$dst/$t.jsonl" ] && continue          # destination started (or has) it: leave it alone
      if [ $dst = dgx ]; then scp -q "$f" "dgx-via-spot3:junyang/carbon/data/bigeval_live/$t.jsonl"
      else scp -q -i $K "$f" "root@${C}:carbon/data/bigeval_live/$t.jsonl"; fi \
        && echo "$(date +%T) $t: $src -> $dst" >> "$L/sync.log"
    done
  done
  sleep "$I"
done
