cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
rm -rf ../data/cnet2_ds_v10
CNET2_WEIGHTED=1 CNET2_UW=1 CNET2_UW_PLANT=1 BUILD_WORKERS=6 ../.venv/bin/python build_v8.py ../data/cnet2_ds_v10 ../data/top_corpus ../data/top_corpus_v6 ../data/ladder_top ../data/wushi_corpus ../data/ours_corpus_v8 > ../data/cnet2_build_v10.log 2>&1
tail -1 ../data/cnet2_build_v10.log
../.venv/bin/python - <<'PY'
import numpy as np, json
D='../data/cnet2_ds_v10/'
w=np.load(D+'w.npy', mmap_mode='r+'); t=np.asarray(np.load(D+'teacher.npy', mmap_mode='r')); s=np.asarray(np.load(D+'step.npy', mmap_mode='r'))
for k,f in ((5,2.0),(6,2.0),(7,2.0),(3,1.5)):
    m=t==k; w[m]=w[m]*f
m=(t==5)&(s<=240); w[m]=w[m]*1.5
m=(t==5)&(s>240);  w[m]=w[m]*0.5
m=np.isin(t,[1,6,7])&(s>200); w[m]=w[m]*1.5
meta=json.load(open(D+'meta.json')); g=np.asarray(np.load(D+'gid.npy', mmap_mode='r'))
hot=np.zeros(len(meta['games']),bool)
for gm in meta['games']:
    for sd in gm['sides']:
        o=gm['names'][1-sd]
        if (o.startswith('无始') or o.startswith('AgyCarbon-P2C')) and gm['scores'][sd]>gm['scores'][1-sd]: hot[gm['gi']]=True
m=hot[g]; w[m]=w[m]*1.5; w.flush()
uw=np.load(D+'uw.npy', mmap_mode='r+'); m=(s>=40)&(s<=100); u=np.asarray(uw[m]); u[:,10:]=np.where(u[:,10:]>1, 3.0, u[:,10:]); uw[m]=u; uw.flush()
print('teacher counts', np.bincount(t, minlength=8).tolist())
PY
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128v9.npz BC_LR=1.5e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v10 ../data/weights/cnet2_b128v10.npz 3 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128v10.log 2>&1
echo TRAIN_DONE
