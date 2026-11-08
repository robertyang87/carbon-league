#!/bin/bash
# stand-in v2 likeness check: v15 (packed) vs Master2s stand-in v2, pure net and with search, 50 games each
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64
while [ ! -f ../data/nocap.done ]; do sleep 20; done
EVO_WORKERS=50 ../.venv/bin/python bigeval.py exp/submission_cnet2_v15.py exp/cnet2_m2s2stand.py 996000 50 c_v15pk_m2s2 >> ../data/m2sval.log 2>&1
EVO_WORKERS=50 ../.venv/bin/python bigeval.py exp/submission_cnet2_v15.py exp/cnet2_m2s2stand_s.py 996000 50 c_v15pk_m2s2s >> ../data/m2sval.log 2>&1
echo DONE > ../data/m2sval.done
