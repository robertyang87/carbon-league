#!/bin/bash
# k4v2r1 calibration double-field, sequential, full-load recipe (EVO_WORKERS=62).
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
echo "CHAIN_START $(date '+%F %T %Z')" > ../data/k4v2r1_chain.log
$PY bigeval.py exp/k4v2r1.py exp/opp_master2r.py 996000 200 k4v2r1_m2 > ../data/k4v2r1_m2.launcher.log 2>&1
echo "M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/k4v2r1_chain.log
$PY bigeval.py exp/k4v2r1.py exp/echo2.py 999000 200 k4v2r1_echo2 > ../data/k4v2r1_echo2.launcher.log 2>&1
echo "ALL_DONE rc=$? $(date '+%F %T %Z')" >> ../data/k4v2r1_chain.log
