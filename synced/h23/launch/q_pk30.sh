#!/bin/bash
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EVO_WORKERS=16 ../.venv/bin/python bigeval.py ../data/packs/snipe30_int8.py exp/opp_master2r.py 994000 200 pk30_m2r > ../data/pk30.out 2>&1
../.venv/bin/python cloud/pack_timing.py ../data/packs/snipe30_int8.py 10 > ../data/pk30_timing.out 2>&1
echo DONE > ../data/pk30.done
