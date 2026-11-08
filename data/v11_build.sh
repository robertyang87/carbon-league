cd /Users/baggioustb/dev/carbon/carbon-league/code
# v18 groundwork (idle CPU only; GPU training waits until v17 is submitted): dataset v11 = v10 + tree-life weights
# + opening weights (teacher sides with >= 6 trees at step 50: their steps <= 60 x2)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
rm -rf ../data/cnet2_ds_v11
nice -n 10 env CNET2_WEIGHTED=1 CNET2_UW=1 CNET2_UW_PLANT=1 CNET2_UW_TREELIFE=1 BUILD_WORKERS=4 ../.venv/bin/python build_v8.py ../data/cnet2_ds_v11 ../data/top_corpus ../data/top_corpus_v6 ../data/ladder_top ../data/wushi_corpus ../data/ours_corpus_v8 > ../data/cnet2_build_v11.log 2>&1
tail -1 ../data/cnet2_build_v11.log
../.venv/bin/python - <<'PY'
import numpy as np, json
D='../data/cnet2_ds_v11/'
w=np.load(D+'w.npy', mmap_mode='r+'); t=np.asarray(np.load(D+'teacher.npy', mmap_mode='r')); s=np.asarray(np.load(D+'step.npy', mmap_mode='r'))
g=np.asarray(np.load(D+'gid.npy', mmap_mode='r'))
for k,f in ((5,2.0),(6,2.0),(7,2.0),(3,1.5)):
    m=t==k; w[m]=w[m]*f
m=(t==5)&(s<=240); w[m]=w[m]*1.5
m=(t==5)&(s>240);  w[m]=w[m]*0.5
m=np.isin(t,[1,6,7])&(s>200); w[m]=w[m]*1.5
meta=json.load(open(D+'meta.json')); ot=json.load(open('../data/opening_trees.json'))
hot=np.zeros(len(meta['games']),bool)
for gm in meta['games']:
    for sd in gm['sides']:
        o=gm['names'][1-sd]
        if (o.startswith('无始') or o.startswith('AgyCarbon-P2C')) and gm['scores'][sd]>gm['scores'][1-sd]: hot[gm['gi']]=True
m=hot[g]; w[m]=w[m]*1.5
# opening weights: rows are side-major per game; a step reset starts the next side
starts=np.r_[0, np.where(np.diff(g)!=0)[0]+1]; ends=np.r_[starts[1:], len(g)]; nup=0
for a,b in zip(starts,ends):
    gm=meta['games'][int(g[a])]; seg=np.concatenate([[0],np.cumsum(np.diff(s[a:b].astype(np.int64))<0)])
    for i,sd in enumerate(gm['sides']):
        if ot.get(gm['key'],{}).get(str(sd), 0) >= 6:
            rr=np.arange(a,b)[(seg==i)&(s[a:b]<=60)]; w[rr]=w[rr]*2.0; nup+=len(rr)
w.flush()
uw=np.load(D+'uw.npy', mmap_mode='r+'); m=(s>=40)&(s<=100); u=np.asarray(uw[m]); u[:,10:]=np.where(u[:,10:]>1, 3.0, u[:,10:]); uw[m]=u; uw.flush()
print('opening rows x2:', nup, 'teacher counts', np.bincount(t, minlength=8).tolist())
PY
echo V11_BUILT
