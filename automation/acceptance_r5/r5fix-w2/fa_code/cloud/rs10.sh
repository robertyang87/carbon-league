#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/rs8.done ]; do sleep 20; done
B=../.venv/bin/python
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000 m2ss:cnet2_m2sstand:996000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/cnet2_v7s_mfxn8_r10.py exp/$f.py $s 200 c_v7s8_r10_$n >> ../data/rs10.log 2>&1
done
echo DONE > ../data/rs10.done
