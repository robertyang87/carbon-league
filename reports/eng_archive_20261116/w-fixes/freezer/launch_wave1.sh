#!/bin/bash
cd ~/workspace/w-fixes/freezer
PY=~/.venvs/k0/bin/python
$PY ~/workspace/v03v05v06-fix/probes/probe_v06b.py src/cnet2_train_fixed.py > evidence/probe_v06b_fixed.log 2>&1
echo "v06b done rc=$?" >> evidence/wave1_status.txt
cd regress/fixture && BC_DEVICE=cpu $PY prove_ties.py > ~/workspace/w-fixes/freezer/evidence/prove_ties_fixed.log 2>&1
echo "ties done rc=$?" >> ~/workspace/w-fixes/freezer/evidence/wave1_status.txt
cd ~/workspace/w-fixes/freezer/regress/root/x/y && $PY -m unittest test_review_fixes -v > ~/workspace/w-fixes/freezer/evidence/test_review_fixes_fixed.log 2>&1
echo "trf done rc=$?" >> ~/workspace/w-fixes/freezer/evidence/wave1_status.txt
