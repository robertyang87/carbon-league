cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CNET2_WEIGHTED=1 CNET2_UW=1 BUILD_WORKERS=6 ../.venv/bin/python build_v6.py ../data/cnet2_ds_v6 ../data/top_corpus ../data/top_corpus_v6 ../data/ladder_top > ../data/cnet2_build_v6.log 2>&1
tail -2 ../data/cnet2_build_v6.log
../.venv/bin/python - <<'PY'
import numpy as np
w=np.load('../data/cnet2_ds_v6/w.npy', mmap_mode='r+'); t=np.asarray(np.load('../data/cnet2_ds_v6/teacher.npy', mmap_mode='r'))
for s,f in ((5,2.0),(6,2.0),(7,2.0),(3,1.5)):
    m=t==s; w[m]=w[m]*f
w.flush(); print('teacher counts', np.bincount(t, minlength=8).tolist())
PY
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128v5.npz BC_REINIT_TEACH=3 BC_LR=3e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v6 ../data/weights/cnet2_b128v6.npz 5 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128v6.log 2>&1
echo TRAIN_DONE
