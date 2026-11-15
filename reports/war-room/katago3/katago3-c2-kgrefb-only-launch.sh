#!/bin/bash
# KataGo probe2 THREE-LEG, C2 — kgref_b ONLY re-run (pre-staged 2026-11-15 ~09:07).
# Context: remain-600 on spot-32 (launched 08:36) lost the machine ~08:53-09:00
# (suspected reclamation) after kgduel_b 200/200 + kgref_a 200/200 landed and
# were verified in the local h32 mirror. kgref_b got 0 lines. This script runs
# ONLY the missing leg on the replacement machine (spot-33 or successor):
# Leg3 ref (2nd seat): Master2 wrapper(2aebfbc1) vs vleaf_value(e8d0b426) -> kgref_b
# PREREG rev2, seeds 985000-985199, subject = vleaf_value e8d0b426, harness 354bcd1b.
# Env/paths identical to katago3-c2-launch.sh. Run ONLY on war-room re-dispatch to a
# provisioned exam machine with the exam tree + KataGo packs installed
# and md5-verified first (field set: bigeval 354bcd1b, eval_v53 45792dfb,
# cnet2_search 94f7fb64, cnet2_np d8c15ad4, fastsim aefe6c83, packs/ x3,
# exp/katago_vleaf_value e8d0b426, katago_vleaf_const 36bbe1e4, echo3_int8 7b762049).
set -u
cd /root/carbon
export EVO_WORKERS=62
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
V=exp/katago_vleaf_value.py
M=exp/opp_master2r.py
date +%s > /root/carbon/katago3_kgrefb_start.txt
$PY code/bigeval.py $M $V 985000 200 kgref_b; echo "RC_ref_b=$?"
date +%s > /root/carbon/katago3_kgrefb_end.txt
echo KATAGO3_KGREFB_ALL_DONE
