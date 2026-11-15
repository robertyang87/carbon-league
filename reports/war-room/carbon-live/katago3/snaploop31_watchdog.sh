#!/bin/bash
# Watchdog for katago3 snaploop31 (C2 field mirror). Re-pulls the loop if it died
# (e.g. VM replacement) while the field is still open. Stops restarting after
# 2026-11-15 07:30 CST: by then the field has moved to a replacement machine
# with its own loop, or the war room handles it manually.
# Only restarts when the local mirror is NOT yet complete for the legs C2 runs
# (attr_b 138 + duel_a 200 + duel_b 200 + ref_a 200 + ref_b 200 = 938 rows).
DIR=/home/hatch/workspace/carbon-live/katago3
OUT=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h31
LOG=/home/hatch/workspace/carbon-live/katago3/snaploop31_watchdog.log
if [ "$(date +%s)" -ge "$(date -d '2026-11-15 07:30:00 +0800' +%s)" ]; then
  echo "$(date '+%F %T %Z') WATCHDOG_EXPIRED" >> "$LOG"
  exit 0
fi
if pgrep -f 'snaploop31\.sh' >/dev/null 2>&1; then
  exit 0
fi
n=0
for t in kgattr_b kgduel_a kgduel_b kgref_a kgref_b; do
  [ -f "$OUT/$t.jsonl" ] && n=$((n + $(wc -l < "$OUT/$t.jsonl")))
done
if [ "$n" -ge 938 ]; then
  echo "$(date '+%F %T %Z') WATCHDOG_DONE rows=$n" >> "$LOG"
  exit 0
fi
cd "$DIR" && setsid nohup ./snaploop31.sh > snaploop31_stdout.log 2>&1 < /dev/null &
echo "$(date '+%F %T %Z') WATCHDOG_RESTART rows=$n pid=$!" >> "$LOG"
