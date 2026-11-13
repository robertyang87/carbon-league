#!/bin/bash
# with_lock: run one command holding the shared .eval_lock (mkdir
# protocol, same as run_gs_stage1.sh). Waits politely, releases on exit.
cd ~/workspace/carbon-league || exit 1
LOCK=.eval_lock
tries=0
while ! mkdir "$LOCK" 2>/dev/null; do
  sleep 15; tries=$((tries+1))
  if [ $tries -gt 120 ]; then echo "LOCK TIMEOUT after $((tries*15))s"; exit 1; fi
done
echo "lock acquired after $((tries*15))s" >&2
trap 'rmdir ~/workspace/carbon-league/.eval_lock 2>/dev/null' EXIT
"$@"
