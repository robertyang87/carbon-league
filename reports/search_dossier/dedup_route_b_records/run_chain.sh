#!/bin/bash
# A/B chain: waits for the foreground probe (dedup_h2 seed 986200) to
# finish, then runs the remaining A/B legs sequentially and compares.
cd ~/workspace/search-group/switch/dedup || exit 1
while pgrep -f "ab_runner.py diag/dedup_h2.py 1 986200" > /dev/null; do sleep 20; done
echo "=== probe done, chain starts $(date) ==="
python3 ab_runner.py diag/dedup_h2.py 1 986201 dedup_h2_b.json 1
python3 - <<'EOF'
import json
a = json.load(open('probe_h2.json')); b = json.load(open('dedup_h2_b.json'))
a['games'] += b['games']
json.dump(a, open('dedup_h2_full.json', 'w'))
print('merged dedup_h2_full games:', len(a['games']))
EOF
python3 ab_runner.py diag/orig_h2.py 2 986200 orig_h2_full.json 0
python3 ab_compare.py orig_h2_full.json dedup_h2_full.json cmp_h2.json
python3 ab_runner.py diag/orig_h1.py 3 986200 orig_h1.json 0
python3 ab_runner.py diag/dedup_h1.py 3 986200 dedup_h1.json 0
python3 ab_compare.py orig_h1.json dedup_h1.json cmp_h1.json
python3 ab_runner.py ../echo4_search_int8.py 2 986200 orig_ship.json 0
python3 ab_runner.py echo4_search_int8_dedup.py 2 986200 dedup_ship.json 0
python3 ab_compare.py orig_ship.json dedup_ship.json cmp_ship.json
echo "CHAIN-DONE $(date)"
