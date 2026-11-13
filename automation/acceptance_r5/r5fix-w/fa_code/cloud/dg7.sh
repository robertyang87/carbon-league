#!/bin/bash
# dg7: dg5 set with old slot-2 (1C-v8) rows moved to slot 1, plus M2 opening rows (steps 0-100) in slot 2 (weight x2).
# Waits for dg6 to finish on the GPU, then trains from dg5 ep5 with slot 2 re-initialised.
cd ~/junyang/carbon/code
D=../data/cnet2_ds_v18dg7
if [ ! -f $D/.ok ]; then
  rm -rf $D
  ../.venv/bin/python concat_ds.py ../data/cnet2_ds_v15dg5 ../data/cnet2_ds_m2early $D 2 || exit 1
  ../.venv/bin/python - <<'PY' || exit 1
import numpy as np
D = "../data/cnet2_ds_v18dg7"; nb = len(np.load("../data/cnet2_ds_v15dg5/y.npy", mmap_mode="r"))
t = np.load(D + "/teacher.npy", mmap_mode="r+")
old = np.asarray(t[:nb]) == 2; t[:nb][old] = 1; t.flush()
c = np.bincount(np.asarray(np.load(D + "/teacher.npy")))
print("slot rows", c.tolist(), "moved", int(old.sum()))
assert c[2] == 6868, c
PY
  touch $D/.ok
fi
while pgrep -f "[c]net2_train.py ../data/cnet2_ds_v17dg6" >/dev/null; do sleep 60; done
BC_DEVICE=cuda BC_RAM=1 BC_THREADS=16 BC_SAVE_EPOCHS=1 BC_COMPILE=1 BC_REINIT_TEACH=2 BC_INIT=../data/weights/cnet2_b128dg5_ep5.npz BC_LR=1.5e-4 \
  ../.venv/bin/python cnet2_train.py $D ../data/weights/cnet2_b128dg7.npz 3 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128dg7.log 2>&1
echo TRAIN_EXIT $? >> ../data/cnet2_train_b128dg7.log
