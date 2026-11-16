#!/bin/bash
cd ~/workspace/w-fixes/freezer
PY=~/.venvs/k0/bin/python
$PY ~/workspace/v03v05v06-fix/probes/probe_v06.py src/cnet2_train_fixed.py > evidence/probe_v06_fixed.log 2>&1
echo "v06 done rc=$?" >> evidence/wave2_status.txt
$PY probes/proof_real_examiner.py src/cnet2_train_fixed.py > evidence/real_examiner/proof.json 2> evidence/real_examiner/proof.err
echo "proof done rc=$?" >> evidence/wave2_status.txt
cd regress/fixture && $PY run_e2e.py > ~/workspace/w-fixes/freezer/evidence/run_e2e_fixed.log 2>&1
echo "e2e done rc=$?" >> ~/workspace/w-fixes/freezer/evidence/wave2_status.txt
