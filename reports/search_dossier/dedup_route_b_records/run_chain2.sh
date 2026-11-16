#!/bin/bash
# A/B chain v2 (post-restart relaunch): all legs sequential, per-game
# incremental JSON dumps, progress lines in this log.
cd ~/workspace/search-group/switch/dedup || exit 1
echo "CHAIN2 START $(date)"
python3 ab_runner.py diag/dedup_h2.py 2 986200 dedup_h2_full.json 0
python3 ab_runner.py diag/orig_h2.py 2 986200 orig_h2_full.json 0
python3 ab_compare.py orig_h2_full.json dedup_h2_full.json cmp_h2.json
python3 ab_runner.py diag/orig_h1.py 2 986200 orig_h1.json 0
python3 ab_runner.py diag/dedup_h1.py 2 986200 dedup_h1.json 0
python3 ab_compare.py orig_h1.json dedup_h1.json cmp_h1.json
python3 ab_runner.py ../echo4_search_int8.py 2 986200 orig_ship.json 0
python3 ab_runner.py echo4_search_int8_dedup.py 2 986200 dedup_ship.json 0
python3 ab_compare.py orig_ship.json dedup_ship.json cmp_ship.json
echo "CHAIN2-DONE $(date)"
