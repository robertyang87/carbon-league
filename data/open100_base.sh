#!/bin/bash
cd /Users/baggioustb/dev/carbon/carbon-league/data
for n in dg5e5s57 dg6s57; do for o in m2s2 v9; do s=$([ $o = m2s2 ] && echo 996000 || echo 997000); bash open100.sh $n $o $s 400; done; done
for o in m2s2 v9; do python3 open100_table.py $o dg5e5s57 dg6s57; done > open100_base.out 2>&1
echo BASE_DONE >> open100.progress
