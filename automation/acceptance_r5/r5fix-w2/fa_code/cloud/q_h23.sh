#!/bin/bash
# host 23: multi-car search (SEQ_KEYS 3) vs Master2 real, 200 games (pairs with m2r_S20m from host 22)
cd /root/carbon/code
until [ -f exp/master2_src.py ] && [ -f exp/fx_m2t3S20q3m.py ]; do sleep 20; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EVO_WORKERS=60 ../.venv/bin/python bigeval.py exp/fx_m2t3S20q3m.py exp/opp_master2r.py 994000 200 m2r_S20q3m > ../data/m2r_S20q3m.out 2>&1
echo DONE > ../data/h23.done
