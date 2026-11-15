#!/bin/bash
# C1 wave-2 returns sync: spot-29 out_wave2 shards -> local mirror -> DGX returns/c1/ (flat).
BASE=/home/hatch/workspace/carbon-live/k0annot
MIR=$BASE/returns_mirror_w2
LOG=$BASE/returns_sync_w2.log
SSH="ssh -F /home/hatch/.ssh/carbon_config"
mkdir -p "$MIR"
pushed=-1
for i in $(seq 1 60); do
  rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_wave2/shards/ "$MIR/" 2>/dev/null
  rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_wave2/progress.json "$MIR/progress_c1_wave2.json" 2>/dev/null
  n=$(find "$MIR" -maxdepth 1 -name "*.npz" | wc -l)
  if [ "$n" != "$pushed" ]; then
    rsync -az --no-owner --no-group -e "$SSH" "$MIR/" dgx:/home/arm/junyang/k0_dist/returns/c1/ 2>/dev/null
    echo "$(date '+%F %T') SYNCED n=$n" >> "$LOG"
    pushed=$n
  else
    echo "$(date '+%F %T') HB n=$n" >> "$LOG"
  fi
  if timeout -k 5 30 ssh -F /home/hatch/.ssh/carbon_config spot29 'test -f /root/carbon/k0annot/wave2_end.txt' 2>/dev/null; then
    rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_wave2/shards/ "$MIR/" 2>/dev/null
    rsync -az --no-owner --no-group -e "$SSH" spot29:/root/carbon/k0annot/out_wave2/progress.json "$MIR/progress_c1_wave2.json" 2>/dev/null
    rsync -az --no-owner --no-group -e "$SSH" "$MIR/" dgx:/home/arm/junyang/k0_dist/returns/c1/ 2>/dev/null
    echo "$(date '+%F %T') WAVE2_DONE_FINAL_SYNCED n=$(find "$MIR" -maxdepth 1 -name "*.npz" | wc -l)" >> "$LOG"
    exit 0
  fi
  sleep 240
done
echo "$(date '+%F %T') LOOP_CAP_EXIT pushed=$pushed" >> "$LOG"
