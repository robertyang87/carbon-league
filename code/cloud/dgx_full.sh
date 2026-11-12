#!/bin/bash
# DGX: whole-network distillation sweep (d0dd809) on cnet2_ds_ex1g after the frozen-trunk ex1g trainings finish.
#
# *** P3 GUARD (REVIEW_FIXES_20261113_R2): UNVERSIONED TRAINER OVERWRITE ***
# The cp below replaces cnet2_train.py with cnet2_train.py.d0dd809 -- a local,
# UNVERSIONED copy that exists only in this working tree (no git blob / hash
# in the repo). After this script runs, the directory's trainer NO LONGER
# matches the versioned repository, and any "core blobs unchanged" conclusion
# stops covering this directory until the trainer is restored from the backup
# (cnet2_train.py.pre_d0dd809, taken by the same cp line). This is a one-way,
# easy-to-miss mutation, so it now requires an explicit opt-in:
#     DGX_FULL_CONFIRM=1 bash dgx_full.sh
# Without DGX_FULL_CONFIRM=1 the script prints a warning and exits BEFORE
# waiting or copying anything.
if [ "${DGX_FULL_CONFIRM:-}" != "1" ]; then
  echo "WARNING: dgx_full.sh would overwrite cnet2_train.py with the UNVERSIONED cnet2_train.py.d0dd809 copy." >&2
  echo "That mutation is not tracked by git and invalidates repo-state claims for this directory." >&2
  echo "Refusing to run. If you really want this, re-run with DGX_FULL_CONFIRM=1." >&2
  exit 1
fi
cd ~/junyang/carbon/code
until grep -q SWEEP_DONE ../data/ex1g_logs/run.log 2>/dev/null; do sleep 30; done
# Guard confirmed above (DGX_FULL_CONFIRM=1): overwrite trainer with the
# unversioned .d0dd809 copy, keeping the pre-overwrite backup alongside.
cp cnet2_train.py cnet2_train.py.pre_d0dd809; cp cnet2_train.py.d0dd809 cnet2_train.py
L=../data/ex1g_logs; PY=../.venv/bin/python
C="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=4 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_USE_W=0 BC_USE_UW=0 BC_EXIT_W=1"
for M in a:300:3e-5 b:1000:3e-5 c:3000:3e-5 d:1000:1e-5; do IFS=: read N K LR <<< "$M"
  echo "== full_$N kl=$K lr=$LR $(date +%T)" >> $L/full.log
  env $C BC_SELF_KL=$K BC_LR=$LR $PY cnet2_train.py ../data/cnet2_ds_ex1g ../data/weights/cnet2_b128ex1f_$N.npz 3 '{"C":128,"NB":8,"D":128}' > $L/train_full_$N.log 2>&1
  grep '"ep"' $L/train_full_$N.log | sed 's/"train_pi.*//' >> $L/full.log
done
echo FULL_DONE >> $L/full.log
