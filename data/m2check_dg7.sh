#!/bin/bash
# A: is dg7's slot 2 M2-like enough to serve as the M2 stand-in? (pure net, slot 2 for all 100 steps, vs the
# Master2s stand-in, 400 games). Pass: t50 >= 5, planted/seized >= 2.5. Pauses the Mac's search games meanwhile.
cd /Users/baggioustb/dev/carbon/carbon-league/data
until [ -f weights/cnet2_b128dg7_ep2.npz ]; do sleep 60; done
pz() { for p in $(pgrep -f "[b]igeval.py exp/submission"); do kill -$1 $p $(pgrep -P $p) 2>/dev/null; done; }
pz STOP
bash open100.sh dg7s257 m2s2 996000 400
pz CONT
python3 open100_table.py m2s2 dg5e5s57 dg7s257 > m2check_dg7.out 2>&1
echo "M2CHECK_DONE" >> open100.progress
