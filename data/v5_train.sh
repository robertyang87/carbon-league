cd /Users/baggioustb/dev/carbon/carbon-league/code
while pgrep -f "build_v5.py" >/dev/null; do sleep 20; done
tail -2 ../data/cnet2_build_v5.log
../.venv/bin/python - <<'PY'
import numpy as np
w=np.load('../data/cnet2_ds_v5/w.npy', mmap_mode='r+'); t=np.load('../data/cnet2_ds_v5/teacher.npy', mmap_mode='r')
new=np.isin(t,[5,6,7]); w[new]=w[new]*2.0; w.flush()
print('teacher counts', np.bincount(np.asarray(t), minlength=8).tolist(), 'new-teacher share', round(float(new.mean()),3))
PY
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128.npz BC_LR=5e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v5 ../data/weights/cnet2_b128v5.npz 6 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128v5.log 2>&1
echo TRAIN_DONE
