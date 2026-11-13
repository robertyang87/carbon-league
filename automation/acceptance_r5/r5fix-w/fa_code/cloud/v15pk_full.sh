#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=24
B=../.venv/bin/python
$B bigeval.py exp/submission_cnet2_v15.py exp/opp_wb532.py 995000 200 c_v15pk_wb532 > /dev/null 2>&1
$B bigeval.py exp/submission_cnet2_v15.py exp/opp_1cz_v9.py 997000 200 c_v15pk_v9 > /dev/null 2>&1
$B bigeval.py exp/submission_cnet2_v15.py exp/cnet2_m2stand.py 994000 50 c_v15pk_m2s > /dev/null 2>&1
$B bigeval.py exp/submission_cnet2_v15.py exp/submission_cnet2_v13.py 993000 200 c_v15pk_v13b > /dev/null 2>&1
$B bigeval.py exp/submission_cnet2_v14.py exp/submission_cnet2_v13.py 993000 200 c_v14pk_v13b > /dev/null 2>&1
echo DONE > ../data/v15pk_full.done
