#!/bin/zsh
# 43-game Master3s: ALL (1 1 1, no anchor, decoder 10 ep), ALLw (1 1 1, whole net lr3e-5 3 ep), P42 (0 1 0, KL30, decoder 10 ep)
cd /Users/baggioustb/dev/carbon/carbon-league/code; D=../data
mk() { X=$D/cnet2_ds_m3s3$1; rm -rf $X; mkdir $X; for f in $D/cnet2_ds_m3s2/*; do [ ${f:t} = uw.npy -o ${f:t} = w.npy ] || ln -s ../cnet2_ds_m3s2/${f:t} $X/${f:t}; done; ../.venv/bin/python set_uw.py $X $2 $3 $4; }
mk all 1 1 1; mk p42 0 1 0
C='{"C":128,"NB":8,"D":128}'; I=$D/weights/cnet2_b128m2t3.npz
DEC="BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1"
env BC_DEVICE=cpu BC_RAM=1 BC_THREADS=3 BC_SAVE_EPOCHS=0 BC_INIT=$I ${(z)DEC} BC_USE_W=1 BC_USE_UW=1 BC_LR=1e-4 ../.venv/bin/python cnet2_train.py $D/cnet2_ds_m3s3all $D/weights/cnet2_b128m3s3all.npz 10 $C > $D/train_m3s3all.log 2>&1 &
env BC_DEVICE=cpu BC_RAM=1 BC_THREADS=3 BC_SAVE_EPOCHS=0 BC_INIT=$I ${(z)DEC} BC_USE_W=1 BC_USE_UW=1 BC_SELF_KL=30 BC_LR=1e-4 ../.venv/bin/python cnet2_train.py $D/cnet2_ds_m3s3p42 $D/weights/cnet2_b128m3s3p42.npz 10 $C > $D/train_m3s3p42.log 2>&1 &
env BC_DEVICE=cpu BC_RAM=1 BC_THREADS=4 BC_SAVE_EPOCHS=0 BC_INIT=$I BC_USE_W=1 BC_USE_UW=1 BC_LR=3e-5 ../.venv/bin/python cnet2_train.py $D/cnet2_ds_m3s3all $D/weights/cnet2_b128m3s3allw.npz 3 $C > $D/train_m3s3allw.log 2>&1 &
wait; ls -l $D/weights/cnet2_b128m3s3*; echo M3S3_TRAINED
