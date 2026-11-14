#!/bin/bash
# KataGo probe2 three-leg SMOKE on spot-29 (C1). 2 games per direction per leg, seeds 985000-985001.
# Legs: attr = vleaf_value vs vleaf_const; duel = vleaf_value vs Echo3 pack; ref = vleaf_value vs Master2 wrapper.
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
echo "SMOKE_START $(date '+%F %T %Z')"
$PY code/bigeval.py $V $C 985000 2 kgsmoke_attr_a; echo "RC_attr_a=$?"
$PY code/bigeval.py $C $V 985000 2 kgsmoke_attr_b; echo "RC_attr_b=$?"
$PY code/bigeval.py $V $E 985000 2 kgsmoke_duel_a; echo "RC_duel_a=$?"
$PY code/bigeval.py $E $V 985000 2 kgsmoke_duel_b; echo "RC_duel_b=$?"
$PY code/bigeval.py $V $M 985000 2 kgsmoke_ref_a; echo "RC_ref_a=$?"
$PY code/bigeval.py $M $V 985000 2 kgsmoke_ref_b; echo "RC_ref_b=$?"
echo "SMOKE_ALL_DONE $(date '+%F %T %Z')"
