#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=16
../.venv/bin/python bigeval.py exp/submission_cnet2_v14p.py exp/submission_cnet2_v13.py 993000 200 c_v14p_v13b > /dev/null 2>&1
echo DONE > ../data/v14p.done
