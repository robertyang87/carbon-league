#!/bin/bash
# foreign-style test: search (packed, no caps) vs v16 net, against opponents whose style is not one of our 8 slots
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/mix.done ]; do sleep 20; done
B=../.venv/bin/python
# 无始 stand-in must pass its ledger check before it is read as a foreign opponent
$B bigeval.py exp/submission_cnet2_v15.py exp/cnet2_wsstand.py 999800 50 c_v15pk_ws >> ../data/foreign.log 2>&1
for o in v5:opp_1cz_v5:999000 ws:cnet2_wsstand:999500; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/cnet2_v7s57_240.py exp/$f.py $s 100 c_v16n_$n >> ../data/foreign.log 2>&1
  $B bigeval.py exp/submission_cnet2_v16s_nocap.py exp/$f.py $s 100 c_v16snc_$n >> ../data/foreign.log 2>&1
done
echo DONE > ../data/foreign.done
