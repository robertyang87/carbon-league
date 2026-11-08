#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
for L in 220 230 250 260; do for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  ../.venv/bin/python bigeval.py exp/cnet2_v6s57_$L.py exp/$f.py $s 200 c_v6s57_${L}_$n > /dev/null 2>&1; done; done
echo DONE > ../data/sweep.done
