#!/bin/bash
# KataGo probe2 THREE-LEG main field, C1/spot-29. PREREG rev2, seeds 985000-985199, double-seat 200 per direction per leg.
# Leg1 attr : vleaf_value(e8d0b426) vs vleaf_const(36bbe1e4)   -> kgattr_a / kgattr_b
# Leg2 duel : vleaf_value vs Echo3 deployment pack(7b762049)  -> kgduel_a / kgduel_b
# Leg3 ref  : vleaf_value vs Master2 wrapper(2aebfbc1)        -> kgref_a / kgref_b
set -u
cd /root/carbon
export EVO_WORKERS=62
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
V=exp/katago_vleaf_value.py
C=exp/katago_vleaf_const.py
E=exp/echo3_int8.py
M=exp/opp_master2r.py
date +%s > /root/carbon/katago3_start.txt
$PY code/bigeval.py $V $C 985000 200 kgattr_a; echo "RC_attr_a=$?"
$PY code/bigeval.py $C $V 985000 200 kgattr_b; echo "RC_attr_b=$?"
$PY code/bigeval.py $V $E 985000 200 kgduel_a; echo "RC_duel_a=$?"
$PY code/bigeval.py $E $V 985000 200 kgduel_b; echo "RC_duel_b=$?"
$PY code/bigeval.py $V $M 985000 200 kgref_a; echo "RC_ref_a=$?"
$PY code/bigeval.py $M $V 985000 200 kgref_b; echo "RC_ref_b=$?"
date +%s > /root/carbon/katago3_end.txt
echo KATAGO3_ALL_DONE
