#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=62
B=../.venv/bin/python
$B bigeval.py exp/cnet2_v7s57_230.py exp/submission_cnet2_v13.py 992000 200 c_v7s57_230_v13 > /dev/null 2>&1
$B bigeval.py exp/cnet2_v7s57_230.py exp/submission_cnet2_v13.py 993000 200 c_v7s57_230_v13b > /dev/null 2>&1
$B bigeval.py exp/cnet2_v7s57_230.py exp/opp_wb532.py 995000 200 c_v7s57_230_wb532 > /dev/null 2>&1
$B bigeval.py exp/cnet2_v7s57_230.py exp/cnet2_m2sstand.py 996000 200 c_v7s57_230_m2ss > /dev/null 2>&1
for c in exp/cnet2_v7s57_240.py exp/submission_cnet2_v15.py; do t=$(basename $c .py); $B bigeval.py $c exp/submission_cnet2_v13.py 998000 400 c_${t}_v13c > /dev/null 2>&1; done
echo DONE > ../data/v7b.done
