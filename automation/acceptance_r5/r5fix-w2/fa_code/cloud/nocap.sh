#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
B=../.venv/bin/python
$B bigeval.py exp/submission_cnet2_v16s_nocap.py exp/submission_cnet2_v13.py 992000 200 c_v16snc_v13 >> ../data/nocap.log 2>&1
$B bigeval.py exp/submission_cnet2_v16s_nocap.py exp/opp_1cz_v9.py 997000 200 c_v16snc_v9 >> ../data/nocap.log 2>&1
$B bigeval.py exp/cnet2_v7s_mfxn8_r10.py exp/opp_1cz_v9.py 997000 200 c_v7s8_r10_v9 >> ../data/nocap.log 2>&1
echo DONE > ../data/nocap.done
