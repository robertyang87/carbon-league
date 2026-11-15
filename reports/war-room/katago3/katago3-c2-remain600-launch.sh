#!/bin/bash
# KataGo probe2 THREE-LEG continuation, C2 — REMAINING 600 games (pre-staged 2026-11-15 ~04:30).
# spot-31 (i-uf6023700gydxbxk4zl6) reclaimed ~03:40 2026-11-15 after 338 lines landed.
# Already saved in exam_snapshots/h31/ (verified: seeds exact, err 0, hh 354bcd1b):
#   kgattr_b 138 (seeds 985062-985199) + kgduel_a 200 (seeds 985000-985199).
# Attribution leg CLOSED (h29 62 + h31 138 = 200). This script runs ONLY the remaining 600:
# Leg2 duel (2nd seat): Echo3 deployment pack(7b762049) vs vleaf_value(e8d0b426) -> kgduel_b
# Leg3 ref  : vleaf_value vs Master2 wrapper(2aebfbc1)                      -> kgref_a / kgref_b
# PREREG rev2, seeds 985000-985199, subject = vleaf_value e8d0b426, harness 354bcd1b.
# Env/paths identical to katago3-c2-launch.sh. Run ONLY on war-room re-dispatch to a
# provisioned exam machine (spot-32 or successor) with the exam tree + KataGo packs installed
# and md5-verified first. Do NOT run on spot-31 (gone) or before the dispatch order.
set -u
cd /root/carbon
export EVO_WORKERS=62
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
V=exp/katago_vleaf_value.py
E=exp/echo3_int8.py
M=exp/opp_master2r.py
date +%s > /root/carbon/katago3_remain600_start.txt
$PY code/bigeval.py $E $V 985000 200 kgduel_b; echo "RC_duel_b=$?"
$PY code/bigeval.py $V $M 985000 200 kgref_a; echo "RC_ref_a=$?"
$PY code/bigeval.py $M $V 985000 200 kgref_b; echo "RC_ref_b=$?"
date +%s > /root/carbon/katago3_remain600_end.txt
echo KATAGO3_REMAIN600_ALL_DONE
