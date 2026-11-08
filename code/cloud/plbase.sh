#!/bin/bash
# planter-ledger baselines with the SAME tool: v16b (uncapped twin) and the v17 candidate on the same stand-in maps
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/v17c.done ]; do sleep 20; done
B=../.venv/bin/python
$B bigeval.py exp/submission_cnet2_v16s_mixnc.py exp/cnet2_m2s2stand.py 996000 200 c_v16b_m2s2_pl >> ../data/plbase.log 2>&1
$B bigeval.py exp/submission_cnet2_v17c_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17c_m2s2_pl >> ../data/plbase.log 2>&1
$B bigeval.py exp/submission_cnet2_v16s_mixnc.py exp/cnet2_wsstand.py 999500 200 c_v16b_ws_pl >> ../data/plbase.log 2>&1
echo DONE > ../data/plbase.done
