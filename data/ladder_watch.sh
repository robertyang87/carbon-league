#!/bin/bash
# every 10 min: per strategy (id:name:tag) ladder record by family + seizures + top-4 + step-150 gap, every 10 new games
cd /Users/baggioustb/dev/carbon/carbon-league/code; export LADDER_BASE=https://117.72.210.134
declare -A last 2>/dev/null
L1=0; L2=0
while true; do
  i=0
  for spec in 072688045615bc3e0f9c59bcccba7218:claude-opus55-cloud-v16b:v16b eeefd6e19021355633d34f6d5a2a1a9b:claude-opus55-cloud-v17:v17; do
    IFS=: read id name tag <<< "$spec"; i=$((i+1))
    python3 ladder.py matches $id ../data/${tag}_matches.json >/dev/null 2>&1
    n=$(python3 -c "import json;print(sum(1 for m in json.load(open('../data/${tag}_matches.json')) if m.get('scores')))" 2>/dev/null)
    prev=$([ $i -eq 1 ] && echo $L1 || echo $L2)
    if [ -n "$n" ] && [ "$n" -ge $((prev+10)) ]; then
      [ $i -eq 1 ] && L1=$n || L2=$n
      mkdir -p ../data/${tag}_ladder/replays; python3 ladder.py replays ../data/${tag}_matches.json ../data/${tag}_ladder/replays >/dev/null 2>&1
      python3 ../data/ladder_summary.py $name $tag 2>&1 | tail -1
    fi
  done
  sleep 600
done
