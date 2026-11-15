#!/bin/bash
# Echo3 deployment package (7b762049) vs Echo2 bridge field, C1/spot-29.
# Seeds 986000-986199, double-seat: run A (Echo3 as pkg A) 200 + run B (Echo3 as pkg B) 200.
set -u
cd /root/carbon
export EVO_WORKERS=62
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
date +%s > /root/carbon/bridge_e3_start.txt
$PY code/bigeval.py exp/echo3_int8.py exp/echo2.py 986000 200 echo3bridge_a
echo "RUN_A_RC=$?"
$PY code/bigeval.py exp/echo2.py exp/echo3_int8.py 986000 200 echo3bridge_b
echo "RUN_B_RC=$?"
date +%s > /root/carbon/bridge_e3_end.txt
echo BRIDGE_ALL_DONE
