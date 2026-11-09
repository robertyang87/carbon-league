#!/bin/bash
# 1) dg5 baselines again with the landing-quality ledger (pc4/sc4); 2) dg7 when its weights reach the Mac:
#    first the M2-likeness check (dg7 slot 2 vs the stand-in: t50, planted/seized, planted nb-carbon), then exam 1.
cd /Users/baggioustb/dev/carbon/carbon-league/data
for o in m2s2 v9; do s=$([ $o = m2s2 ] && echo 996000 || echo 997000); bash open100.sh dg5e5s57 $o $s 400; done
until [ -f weights/cnet2_b128dg7_ep2.npz ]; do sleep 60; done
bash open100.sh dg7s257 m2s2 996000 400
python3 open100_table.py m2s2 dg5e5s57 dg7s257 > open100_dg7_m2check.out 2>&1; echo "M2CHECK_DONE" >> open100.progress
bash open100.sh dg7s257 v9 997000 400
for o in m2s2 v9; do s=$([ $o = m2s2 ] && echo 996000 || echo 997000); bash open100.sh dg7s57 $o $s 400; done
for o in m2s2 v9; do python3 open100_table.py $o dg5e5s57 dg7s57 dg7s257; done > open100_dg7.out 2>&1
echo DG7_DONE >> open100.progress
