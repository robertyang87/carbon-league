#!/bin/bash
# Stable big-model recipe for cnet2 (run from code/). GPU run:  ./cnet2_big_recipe.sh gpu
# CPU sanity (tiny subset, 1 epoch):                           ./cnet2_big_recipe.sh sanity
# Changes vs the diverged b160 runs: fp32 (no bf16 autocast), zero-init residual branch (ZI=1: every block
# starts as identity), 15% linear-ish warmup (OneCycle pct_start), peak lr 5e-4, grad clip 1.0.
CFG='{"C":160,"NB":10,"D":160,"ZI":1}'
DATA=${DATA:-../data/cnet2_ds_v4}
if [ "$1" = "sanity" ]; then
  BC_DEVICE=cpu BC_AMP=fp32 BC_SUBSET=3000 BC_BS=128 BC_LR=5e-4 BC_PCT=0.15 BC_CLIP=1.0 OMP_NUM_THREADS=2 \
    ../.venv/bin/python cnet2_train.py $DATA ${SANITY_OUT:-../data/weights/cnet2_b160s_sanity.npz} 1 "$CFG"
else
  BC_DEVICE=mps BC_AMP=fp32 BC_BS=1024 BC_LR=5e-4 BC_PCT=0.15 BC_CLIP=1.0 \
    nohup ../.venv/bin/python cnet2_train.py $DATA ../data/weights/cnet2_b160s.npz ${EPOCHS:-16} "$CFG" \
    > ../data/cnet2_train_b160s.log 2>&1 &
  echo $! > ../data/cnet2_train_b160s.pid
fi
