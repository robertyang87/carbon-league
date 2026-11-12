#!/bin/bash
# DGX: whole-network distillation sweep (d0dd809) on cnet2_ds_ex1g after the frozen-trunk ex1g trainings finish.
cd ~/junyang/carbon/code
until grep -q SWEEP_DONE ../data/ex1g_logs/run.log 2>/dev/null; do sleep 30; done
cp cnet2_train.py cnet2_train.py.pre_d0dd809; cp cnet2_train.py.d0dd809 cnet2_train.py
L=../data/ex1g_logs; PY=../.venv/bin/python
C="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=4 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_USE_W=0 BC_USE_UW=0 BC_EXIT_W=1"
for M in a:300:3e-5 b:1000:3e-5 c:3000:3e-5 d:1000:1e-5; do IFS=: read N K LR <<< "$M"
  echo "== full_$N kl=$K lr=$LR $(date +%T)" >> $L/full.log
  env $C BC_SELF_KL=$K BC_LR=$LR $PY cnet2_train.py ../data/cnet2_ds_ex1g ../data/weights/cnet2_b128ex1f_$N.npz 3 '{"C":128,"NB":8,"D":128}' > $L/train_full_$N.log 2>&1
  grep '"ep"' $L/train_full_$N.log | sed 's/"train_pi.*//' >> $L/full.log
done
echo FULL_DONE >> $L/full.log
