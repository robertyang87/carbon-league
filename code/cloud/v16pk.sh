#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
../.venv/bin/python bigeval.py exp/submission_cnet2_v16s.py exp/submission_cnet2_v13.py 992000 200 c_v16spk_v13 > ../data/v16pk.out 2>&1; echo DONE > ../data/v16pk.done
