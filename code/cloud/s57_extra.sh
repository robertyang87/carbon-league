#!/bin/bash
cd ~/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=24
../.venv/bin/python bigeval.py exp/cnet2_v6s57_240.py exp/opp_1cz_v9.py 997000 200 c_v6s57_240_v9 > /dev/null 2>&1
../.venv/bin/python bigeval.py exp/cnet2_v6s57_240.py exp/cnet2_m2stand.py 994000 50 c_v6s57_240_m2s > /dev/null 2>&1
echo EXTRA_DONE > ../data/s57_extra.done
