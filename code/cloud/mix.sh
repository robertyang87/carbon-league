#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64
while [ ! -f ../data/f16.done ]; do sleep 20; done
EVO_WORKERS=60 ../.venv/bin/python bigeval.py exp/submission_cnet2_v16s_mixnc.py exp/submission_cnet2_v13.py 992000 200 c_v16smix_v13 >> ../data/mix.log 2>&1
echo DONE > ../data/mix.done
