#!/bin/bash
# host 10: first whole-game AWR round without the anchor (anchor set only on the DGX). Baseline at lr 0, then lr 1e-4 x 2.
cd /root/carbon/code; L=/root/carbon/data/train_sp1.log
export OMP_NUM_THREADS=60 OPENBLAS_NUM_THREADS=60 MKL_NUM_THREADS=60
until /root/carbon/.venv/bin/python -c "import torch" 2>/dev/null; do sleep 20; done
P=/root/carbon/.venv/bin/python; D=../data/cnet2_ds_sp1; SRC="../data/sp1_m2s ../data/sp1_m2so ../data/sp1_scout1"
{ echo "== torch $($P -c 'import torch; print(torch.__version__, torch.get_num_threads())')"; date
  echo "== build"; CNET2_STEP_MAX=300 BUILD_WORKERS=60 $P build_awr.py $D $SRC 2>&1 | tail -5
  echo "== weights"; $P awr_weights.py $D $SRC 2>&1 | tail -8
  T="BC_DEVICE=cpu BC_RAM=1 BC_THREADS=60 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1"
  echo "== baseline lr 0"; date; env $T BC_LR=0 $P cnet2_train.py $D ../data/weights/cnet2_b128sp1_lr0.npz 1 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -12
  echo "== train lr 1e-4 x2"; date; env $T BC_LR=1e-4 $P cnet2_train.py $D ../data/weights/cnet2_b128sp1.npz 2 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -14
  date; echo TRAIN_DONE; } > $L 2>&1
