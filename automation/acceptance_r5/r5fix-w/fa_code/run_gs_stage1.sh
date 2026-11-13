#!/bin/bash
# GS stage 1: wait for the shared eval lock, then seize/guard variants
# vs switcher_harvest (4 games each) + vs base_v53 side-effect check
# (4 games each), then record traces. <=2 concurrent games throughout
# (ProcessPoolExecutor max_workers=2 inside the eval/record scripts).
cd ~/workspace/carbon-league || exit 1
LOCK=.eval_lock
tries=0
while ! mkdir "$LOCK" 2>/dev/null; do
  sleep 15; tries=$((tries+1))
  if [ $tries -gt 240 ]; then echo "LOCK TIMEOUT"; exit 1; fi
done
echo "lock acquired after $((tries*15))s"
trap 'rmdir ~/workspace/carbon-league/.eval_lock 2>/dev/null' EXIT
for ph in seize_h guard_h seize_base guard_base; do
  echo "=== eval $ph ==="
  EVAL_WORKERS=2 python3 eval_gs.py $ph
done
echo "=== record traces ==="
python3 record_gs_traces.py gs_seize_h gs_guard_h gs_seize_base gs_guard_base 2>&1 | tail -30
echo "=== GS STAGE1 DONE ==="
