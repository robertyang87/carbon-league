import json, os, sys, hashlib
import numpy as np
SRC='/home/arm/junyang/carbon/data/ds_m9xj'; DST='/home/arm/junyang/carbon/data/ds_m9xj_w'
# Stacking rule (stated): factors MULTIPLY. Train row factor = 3 if step>=270,
# times 2 if its side lost the game or the game margin (either side) is <=500.
# A row hit by both rules appears 6 times. Validation games: factor 1, untouched.
meta=json.load(open(os.path.join(SRC,'meta.json'))); games=meta['games']
gid=np.load(os.path.join(SRC,'gid.npy')); step=np.load(os.path.join(SRC,'step.npy'))
N=len(gid); factors=np.ones(N,np.int64)
rep={'end_only':0,'lc_only':0,'both':0,'plain':0,'val_rows':0,'train_rows':0,
     'end_rows_train':0,'lost_games':0,'close_games':0,'val_games':0}
is_val=np.zeros(N,bool)
for gi,g in enumerate(games):
    idx=np.nonzero(gid==gi)[0]
    if len(idx)==0: continue
    if g['val']:
        is_val[idx]=True; rep['val_rows']+=len(idx); rep['val_games']+=1; continue
    rep['train_rows']+=len(idx)
    st=step[idx].astype(np.int64)
    seg=np.concatenate([[0],np.cumsum(np.diff(st)<0)])
    sides=g['sides']; scores=g['scores']
    gmargin=abs(float(scores[0])-float(scores[1]))
    side_margins={s: float(scores[s])-float(scores[1-s]) for s in sides}
    if any(m<0 for m in side_margins.values()): rep['lost_games']+=1
    if gmargin<=500: rep['close_games']+=1
    for j,i in enumerate(idx):
        s=sides[int(seg[j])]; end=bool(st[j]>=270)
        lc=(side_margins[s]<0) or (gmargin<=500)
        if end: rep['end_rows_train']+=1
        f=(3 if end else 1)*(2 if lc else 1); factors[i]=f
        if end and lc: rep['both']+=1
        elif end: rep['end_only']+=1
        elif lc: rep['lc_only']+=1
        else: rep['plain']+=1
repidx=np.repeat(np.arange(N),factors); NN=len(repidx)
os.makedirs(DST,exist_ok=True)
fields=sorted(f[:-4] for f in os.listdir(SRC) if f.endswith('.npy'))
for k in fields:
    a=np.load(os.path.join(SRC,k+'.npy')); np.save(os.path.join(DST,k+'.npy'), a[repidx])
# meta: same games/val/teachers; per-game n and total updated to expanded counts
newn=np.bincount(gid[repidx], minlength=len(games))
meta2=json.loads(json.dumps(meta)); meta2['total']=int(NN)
for gi,g in enumerate(meta2['games']): g['n']=int(newn[gi])
meta2['weighted']={'from':'ds_m9xj','rule':'train rows: x3 if step>=270, x2 if side lost or |game margin|<=500, multiplicative (both=x6); val games untouched'}
json.dump(meta2, open(os.path.join(DST,'meta.json'),'w'))
# validation: val rows byte-identical (same order, same bytes) in every field
gid2=np.load(os.path.join(DST,'gid.npy'))
valg=np.array([g['val'] for g in games]); is_val2=valg[gid2]
assert int(is_val2.sum())==int(is_val.sum()), 'val row count changed'
for k in fields:
    a=np.load(os.path.join(SRC,k+'.npy'))[is_val]; b=np.load(os.path.join(DST,k+'.npy'))[is_val2]
    assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b), 'val mismatch field '+k
    h=hashlib.md5(b.tobytes()).hexdigest()
    if k in ('y','step','gid'): rep[f'val_md5_{k}']=h
rep['rows_before']=int(N); rep['rows_after']=int(NN)
rep['train_rows_after']=int(NN-rep['val_rows'])
rep['end_rows_train_after']=int(rep['end_only']*3+rep['both']*6)
print('WEIGHTED-REPORT', json.dumps(rep))
print('VAL-IDENTICAL-OK fields=', len(fields))
