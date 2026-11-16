#!/bin/bash
cd ~/workspace/r4-fixes/x07/stage
CANON=/home/hatch/workspace/heads-v2-diff/dgx.npz
FIX=~/workspace/r4-fixes/x07/fixtures
EV=~/workspace/r4-fixes/x07/evidence
echo "== A: student=corrupt_nan parent=corrupt_nan (BASE driver) =="
python3 k0_duel.py --student $FIX/corrupt_nan.npz --parent $FIX/corrupt_nan.npz \
  --roots $FIX/roots_x07.json --out $EV/reproA_base.json
echo "A exit=$?"
echo "== B: student=corrupt_nan parent=canonical (BASE driver) =="
python3 k0_duel.py --student $FIX/corrupt_nan.npz --parent $CANON \
  --roots $FIX/roots_x07.json --out $EV/reproB_base.json
echo "B exit=$?"
echo "== C: student=corrupt_inf parent=corrupt_inf (BASE driver) =="
python3 k0_duel.py --student $FIX/corrupt_inf.npz --parent $FIX/corrupt_inf.npz \
  --roots $FIX/roots_x07.json --out $EV/reproC_base.json
echo "C exit=$?"
echo REPRO_DONE
