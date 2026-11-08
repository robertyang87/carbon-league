#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=24
while [ ! -f ../data/v15pk_full.done ]; do sleep 20; done
B=../.venv/bin/python
$B bigeval.py exp/submission_cnet2_v14.py exp/cnet2_m2sstand.py 996000 200 c_v14pk_m2ss > /dev/null 2>&1
$B bigeval.py exp/submission_cnet2_v15.py exp/cnet2_m2sstand.py 996000 200 c_v15pk_m2ss > /dev/null 2>&1
echo DONE > ../data/m2s_exam.done
