#!/bin/bash
# S3 v2 batch: wait for the shared eval lock, then sanity -> main ->
# abl -> trace recording, then release. All games <=2 concurrent
# (ProcessPoolExecutor max_workers=2 inside the eval/record scripts).
cd ~/workspace/carbon-league || exit 1
LOCK=.eval_lock
tries=0
while ! mkdir "$LOCK" 2>/dev/null; do
  sleep 20; tries=$((tries+1))
  if [ $tries -gt 180 ]; then echo "LOCK TIMEOUT"; exit 1; fi
done
echo "lock acquired after $((tries*20))s"
trap 'rmdir ~/workspace/carbon-league/.eval_lock 2>/dev/null' EXIT
echo "=== sanity ==="
python3 eval_s3p2.py sanity
echo "=== main ==="
python3 eval_s3p2.py main
echo "=== abl ==="
python3 eval_s3p2.py abl
echo "=== record traces ==="
python3 record_s3p2_traces.py 2>&1 | tail -30
echo "=== BATCH DONE ==="
