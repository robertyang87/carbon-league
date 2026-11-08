#!/bin/bash
# every 10 min: v16b ladder record by family + seizures + platform timing; one line per new batch of games
cd /Users/baggioustb/dev/carbon/carbon-league/code; export LADDER_BASE=https://117.72.210.134
last=0
while true; do
  python3 ladder.py matches 072688045615bc3e0f9c59bcccba7218 ../data/v16b_matches.json >/dev/null 2>&1
  n=$(python3 -c "import json;print(sum(1 for m in json.load(open('../data/v16b_matches.json')) if m.get('scores')))" 2>/dev/null)
  if [ -n "$n" ] && [ "$n" -ge $((last+10)) ]; then
    last=$n
    python3 ladder.py replays ../data/v16b_matches.json ../data/v16b_ladder/replays >/dev/null 2>&1
    python3 ../data/v16b_summary.py 2>&1 | tail -1
  fi
  sleep 600
done
