#!/bin/bash
# GS stage 2: same-convention control (base_v53 as policy vs
# switcher_harvest, seeds 1000-1001) to adjudicate the one slot where
# both variants deviated identically from the cross-convention chk
# baseline; then record traces. <=2 concurrent games.
cd ~/workspace/carbon-league || exit 1
LOCK=.eval_lock
tries=0
while ! mkdir "$LOCK" 2>/dev/null; do
  sleep 15; tries=$((tries+1))
  if [ $tries -gt 240 ]; then echo "LOCK TIMEOUT"; exit 1; fi
done
echo "lock acquired after $((tries*15))s"
trap 'rmdir ~/workspace/carbon-league/.eval_lock 2>/dev/null' EXIT
echo "=== eval base_h1 ==="
EVAL_WORKERS=2 python3 eval_gs.py base_h1
echo "=== record traces ==="
python3 record_gs_traces.py gs_base_h1 2>&1 | tail -10
echo "=== GS STAGE2 DONE ==="
