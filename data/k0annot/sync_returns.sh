#!/bin/bash
# C1: sync landed shards + progress.json copy to DGX returns/c1/ (run on spot-29).
set -u
ROOT=/root/carbon/k0annot
DEST=/home/arm/junyang/k0_dist/returns/c1
rsync -az --no-owner --no-group -e "ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  "$ROOT/out_timing/shards/" "arm@100.84.148.73:$DEST/shards/" 2>/dev/null
rsync -az --no-owner --no-group -e "ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  "$ROOT/out_timing/progress.json" "arm@100.84.148.73:$DEST/progress_c1.json" 2>/dev/null
date "+%F %T sync done" >> "$ROOT/sync_returns.log"
