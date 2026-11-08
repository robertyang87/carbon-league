#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=24
../.venv/bin/python bigeval.py exp/submission_cnet2_v15.py exp/submission_cnet2_v13.py 992000 200 c_v15pk_v13 > /dev/null 2>&1
echo DONE > ../data/v15pk.done
