#!/bin/bash
# parity4.sh NEW.py REF.py SEED0 N  -- game-for-game parity (fixed-budget packs) in 4 parallel chunks on this machine.
# Prints how many games had identical actions and final cash; any non-SAME line is shown.
cd "$(dirname "$0")/.."; export OMP_NUM_THREADS=1
n=$(( $4 / 4 )); T=$(mktemp -d "${TMPDIR:-/tmp}/par.XXXX")
for i in 0 1 2 3; do ../.venv/bin/python cnet2_search_parity.py "$1" "$2" exp/cnet2_m2s2stand.py $(( $3 + i * n )) $n > "$T/$i.txt" 2>&1 & done; wait
same=$(cat "$T"/*.txt | grep -c " SAME$"); total=$(( n * 4 ))
echo "PARITY games SAME: $same/$total"
cat "$T"/*.txt | grep -v " SAME$" | grep -v "ALL SAME" | head -5
rm -rf "$T"
[ "$same" = "$total" ]
