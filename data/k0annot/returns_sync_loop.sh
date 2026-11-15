#!/bin/bash
# C1 returns sync: spot-29 shards -> local mirror -> DGX returns/c1/ (flat).
BASE=/home/hatch/workspace/carbon-live/k0annot
MIR=$BASE/returns_mirror
LOG=$BASE/returns_sync.log
SSH="ssh -F /home/hatch/.ssh/carbon_config"
mkdir -p "$MIR"
pushed=-1
for i in $(seq 1 45); do
  rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_timing/shards/ "$MIR/" 2>/dev/null
  rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_timing/progress.json "$MIR/progress_c1.json" 2>/dev/null
  n=$(find "$MIR" -maxdepth 1 -type f ! -name "progress_c1.json" | wc -l)
  if [ "$n" != "$pushed" ]; then
    rsync -az --no-owner --no-group -e "$SSH" "$MIR/" dgx:/home/arm/junyang/k0_dist/returns/c1/ 2>/dev/null
    if [ "$pushed" = "-1" ] && [ "$n" -ge 1 ]; then
      echo "$(date '+%F %T') FIRST_SHARD_SYNCED n=$n" >> "$LOG"
    else
      echo "$(date '+%F %T') SYNCED n=$n" >> "$LOG"
    fi
    pushed=$n
  fi
  if timeout -k 5 30 ssh -F /home/hatch/.ssh/carbon_config spot29 'test -f /root/carbon/k0annot/wave1_end.txt' 2>/dev/null; then
    rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_timing/shards/ "$MIR/" 2>/dev/null
    rsync -az --no-owner --no-group -e "$SSH" "$MIR/" dgx:/home/arm/junyang/k0_dist/returns/c1/ 2>/dev/null
    echo "$(date '+%F %T') WAVE1_DONE_FINAL_SYNCED n=$(find "$MIR" -maxdepth 1 -type f ! -name "progress_c1.json" | wc -l)" >> "$LOG"
    exit 0
  fi
  sleep 240
done
echo "$(date '+%F %T') LOOP_CAP_EXIT pushed=$pushed" >> "$LOG"
