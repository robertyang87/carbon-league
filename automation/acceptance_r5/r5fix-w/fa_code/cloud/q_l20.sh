#!/bin/bash
# L20 GPU (2026-11-11 01:4x): same first AWR round as q_h11.sh, trained on CUDA. Data uploaded from the Mac; DATA_READY marks it complete.
cd /root/carbon/code; L=/root/carbon/data/train_sp1.log; P=/root/carbon/.venv/bin/python
{ echo "== torch install $(date +%T)"; $P -m pip uninstall -y -q torch 2>/dev/null; $P -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
  export OMP_NUM_THREADS=60 OPENBLAS_NUM_THREADS=60 MKL_NUM_THREADS=60
  echo "== torch $($P -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))')"
  until [ -f /root/carbon/data/DATA_READY ]; do sleep 15; done
  D=../data/cnet2_ds_sp1; SRC="../data/sp1_m2s ../data/sp1_m2so ../data/sp1_scout1"; rm -rf $D
  echo "== build $(date +%T)"; CNET2_STEP_MAX=300 BUILD_WORKERS=60 $P build_awr.py $D $SRC 2>&1 | tail -5
  echo "== weights $(date +%T)"; $P awr_weights.py $D $SRC 2>&1 | tail -8
  T="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=16 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1"
  echo "== baseline lr 0 $(date +%T)"; env $T BC_LR=0 $P cnet2_train.py $D ../data/weights/cnet2_b128sp1_lr0.npz 1 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -12
  echo "== train lr 1e-4 x2 $(date +%T)"; env $T BC_LR=1e-4 $P cnet2_train.py $D ../data/weights/cnet2_b128sp1.npz 2 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -14
  date; echo TRAIN_DONE; } > $L 2>&1
