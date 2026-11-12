#!/bin/bash
# DGX GPU sweep while the CPU runs exams: build cnet2_ds_<TAG> from ../data/<TAG>_in/*/ex1_* (4 build workers, 4 threads),
# then train each mix "NAME:EXIT_W:SELF_KL:EPOCHS" and log changed_learned / unchanged_kept / collector per epoch.
cd ~/junyang/carbon/code
PY=../.venv/bin/python; G=$1; shift; L=../data/${G}_logs; mkdir -p $L
DIRS=$(ls -d ../data/${G}_in/*/ex1_* | grep -v '\.out$' | tr '\n' ' ')
if [ ! -d ../data/cnet2_ds_$G ]; then
  echo "== build $(date +%T) $DIRS" >> $L/run.log
  CNET2_STEP_MAX=239 BUILD_WORKERS=4 $PY build_awr.py ../data/cnet2_ds_$G $DIRS > $L/build.log 2>&1 || { echo BUILD_FAIL >> $L/run.log; exit 1; }
  tail -1 $L/build.log >> $L/run.log
fi
T="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=4 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1 BC_USE_W=0 BC_USE_UW=0"
for M in "$@"; do IFS=: read N W K E <<< "$M"
  echo "== $N exit_w=$W kl=$K ep=$E $(date +%T)" >> $L/run.log
  env $T BC_SELF_KL=$K BC_EXIT_W=$W BC_LR=$( [ "$W" = 0 ] && echo 0 || echo 1e-4 ) $PY cnet2_train.py ../data/cnet2_ds_$G ../data/weights/cnet2_b128${G}$N.npz $E '{"C":128,"NB":8,"D":128}' > $L/train_$N.log 2>&1
  grep '"ep"' $L/train_$N.log | sed 's/"train_pi.*//' >> $L/run.log
done
echo SWEEP_DONE >> $L/run.log
