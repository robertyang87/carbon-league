#!/bin/bash
# after te.sh: v17tb (go-home rule + ambush-aware base choice) on the same maps as c_v17t_v9 / c_v17te_v9
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/te.done ]; do sleep 30; done
B=../.venv/bin/python
$B bigeval.py exp/submission_cnet2_v17tb_nc.py exp/opp_1cz_v9.py 997000 200 c_v17tb_v9 >> ../data/te2.log 2>&1; echo "$(date +%T) v17tb v9" >> ../data/te.progress
echo DONE > ../data/te2.done
