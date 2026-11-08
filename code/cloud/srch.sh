#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=62
while [ ! -f ../data/v7b.done ]; do sleep 20; done
../.venv/bin/python bigeval.py exp/cnet2_v7s_mfxn.py exp/submission_cnet2_v13.py 992000 200 c_v7s_mfxn_v13 > ../data/srch_v13.out 2>&1
echo DONE > ../data/srch.done
