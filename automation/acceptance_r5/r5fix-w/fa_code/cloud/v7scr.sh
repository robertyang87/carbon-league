#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
for c in v7s57_240 v7t5g v7t7g; do
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/submission_cnet2_v13.py 992000 200 c_${c}_v13 > /dev/null 2>&1
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/submission_cnet2_v13.py 993000 200 c_${c}_v13b > /dev/null 2>&1
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/opp_wb532.py 995000 200 c_${c}_wb532 > /dev/null 2>&1
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/cnet2_m2sstand.py 996000 200 c_${c}_m2ss > /dev/null 2>&1
done
echo DONE > ../data/v7scr.done
