#!/bin/bash
# Studio ppou2 probe launch driver. Runs ON the Studio.
# Pre-check (orphan-worker gate): no rl2_ppo_train process may be running.
# Then launches the probe launcher copy, starts probe_watch.py, verifies.
set -u
P=/Users/arm/junyang/carbon/data/ppou2_probe_studio
PY=/Users/arm/junyang/carbon/.venv/bin/python
START=$(date +%s)
echo "START_EPOCH $START $(date '+%F %T %Z')"
echo "---PRECHECK---"
if pgrep -f 'rl2_ppo_trai[n]' >/dev/null 2>&1; then
  echo "PRECHECK_FAIL: rl2_ppo_train already running"
  pgrep -fl 'rl2_ppo_trai[n]' || true
  exit 1
fi
echo "PRECHECK_OK: no train process"
echo "---LAUNCH---"
$PY $P/launch_ppou2_v7_probe.py m --resume $P/state_frozen_it40.pt
echo "---LAUNCH-JSON---"
cat $P/out/ppou2_probe.launch.json
PGID=$(/usr/bin/python3 -c "import json;print(json.load(open('$P/out/ppou2_probe.launch.json'))['pgid'])")
echo "PGID=$PGID"
nohup /usr/bin/python3 $P/probe_watch.py "$PGID" "$START" > $P/out/probe_watch.log 2>&1 < /dev/null &
echo "WATCHER_PID=$!"
sleep 25
echo "---GROUP---"
ps -ax -o pgid= -o pid= -o etime= -o rss= -o command= | awk -v g="$PGID" '$1==g' | head -25
echo "---LOG-HEAD---"
head -c 2000 $P/out/ppou2_probe.log
echo "---LOG-TAIL---"
tail -c 2000 $P/out/ppou2_probe.log
echo "---OUT-LS---"
ls -la $P/out/
