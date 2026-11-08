cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CNET2_WEIGHTED=1 CNET2_UW=1 BUILD_WORKERS=6 ../.venv/bin/python build_v6.py ../data/cnet2_ds_v7 ../data/top_corpus ../data/top_corpus_v6 ../data/ladder_top > ../data/cnet2_build_v7.log 2>&1
tail -1 ../data/cnet2_build_v7.log
../.venv/bin/python - <<'PY'
import numpy as np
D='../data/cnet2_ds_v7/'
w=np.load(D+'w.npy', mmap_mode='r+'); t=np.asarray(np.load(D+'teacher.npy', mmap_mode='r')); s=np.asarray(np.load(D+'step.npy', mmap_mode='r'))
for k,f in ((5,2.0),(6,2.0),(7,2.0),(3,1.5)):          # v6 teacher bumps
    m=t==k; w[m]=w[m]*f
# phase-aware teachers: Master2 economy early, 1C endgame late
m=(t==5)&(s<=240); w[m]=w[m]*1.5
m=(t==5)&(s>240);  w[m]=w[m]*0.5
m=np.isin(t,[1,6,7])&(s>200); w[m]=w[m]*1.5
w.flush(); print('teacher counts', np.bincount(t, minlength=8).tolist())
PY
uw=../data/cnet2_ds_v7/uw.npy
../.venv/bin/python - <<'PY'
import numpy as np
D='../data/cnet2_ds_v7/'
uw=np.load(D+'uw.npy', mmap_mode='r+'); s=np.asarray(np.load(D+'step.npy', mmap_mode='r'))
m=(s>=40)&(s<=100); u=np.asarray(uw[m]); u[:,10:]=np.where(u[:,10:]>1, 3.0, u[:,10:]); uw[m]=u; uw.flush()   # base recruit 40-100: x2 -> x3
print('uw base 40-100 set to 3')
PY
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128v6.npz BC_LR=2e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v7 ../data/weights/cnet2_b128v7.npz 4 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128v7.log 2>&1
echo TRAIN_DONE
