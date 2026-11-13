#!/bin/bash
# DGX: pure-net exams vs 1C-v9 (seeds 997000, 200 games, ledger lx6), run as each net's training finishes.
cd /home/arm/junyang/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=12
B=../.venv/bin/python; P=../data/v9econ.progress
ex() { local f=../data/bigeval_live/v9_$1_lx6.jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 200 ] && return; rm -f $f
  $B bigeval.py exp/cnet2_$1.py exp/opp_1cz_v9.py 997000 200 v9_$1_lx6 >> ../data/v9econ.log 2>&1; echo "$(date +%T) $1 $(wc -l < $f)" >> $P; }
ex dg5e5s57
until grep -q TRAIN_DONE ../data/cnet2_train_b128dg6.log 2>/dev/null; do sleep 60; done; ex dg6s57
until grep -q "TRAIN_EXIT" ../data/cnet2_train_b128dg7.log 2>/dev/null; do sleep 60; done; ex dg7s257; ex dg7s57
echo DONE >> $P
