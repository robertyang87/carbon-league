#!/bin/bash
# L20 (2026-11-11 02:3x): round 1 again with the collector anchor (awr_collector_anchor.py), same data and settings.
cd /root/carbon/code; L=/root/carbon/data/train_sp1b.log; P=/root/carbon/.venv/bin/python
{ D=../data/cnet2_ds_sp1b; rm -rf $D; $P awr_collector_anchor.py ../data/cnet2_ds_sp1 $D 1
  T="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=16 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1"
  echo "== train lr 1e-4 x2 + collector anchor $(date +%T)"; env $T BC_LR=1e-4 $P cnet2_train.py $D ../data/weights/cnet2_b128sp1b.npz 2 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -8
  date; echo TRAIN_DONE; } > $L 2>&1
