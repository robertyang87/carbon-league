"""Audit split-wise support, read full cash from official frames and independently re-sum the credited current trees."""
from pathlib import Path
import gzip,json
import numpy as np
from labels import HERE,BROAD,CACHE,read,sha
from pydantic_core import from_json
def run():
    panel=read(BROAD/'panel.json')['games'];out={}
    with np.load(HERE/'labels_25.npz',allow_pickle=False) as z:
        for split in ('train','val','holdout'):
            cards=[g for g in panel if g['split']==split]
            b=np.isin(z['gid'],[g['gid'] for g in cards]);h=z['horizon'][b];v=z['target'][b]
            nz=np.count_nonzero(v,axis=0);assert (nz>20).all() and (v[:,1:]>=-.01).all()
            out[split]=dict(maps=len(cards),roots=len(v),full25=int((h==25).sum()),min=int(h.min()),max=int(h.max()),
                 nonzero=nz.tolist(),median=np.median(v,axis=0).tolist(),p95=np.quantile(v,.95,axis=0).tolist())
        # Use a stable ten-map stratification, no choice by absolute rewards or outcome.
        checks=[]
        for teacher in sorted({g['teacher'] for g in panel}):
            for seat in (0,1):
                g=next(g for g in panel if g['seat']==seat and g['teacher']==teacher and g['split']=='train')
                doc=from_json(gzip.decompress((CACHE/(g['id']+'.json.gz')).read_bytes()))
                rows=z['gid']==g['gid'];v=z['target'][rows];steps=z['step'][rows].astype(int);h=z['horizon'][rows]
                worlds=[ff[0]['observation'] for ff in doc['steps']];money=[[w['players'][s][0] for s in (seat,1-seat)] for w in worlds]
                pay=np.array([[sum(w['trees'][tree][1] for tree in w['players'][s][3] if tree in w['trees']) for s in (seat,1-seat)] for w in worlds])
                cash=np.asarray(money)
                diff=(cash[steps+h]-cash[steps]);expected=np.column_stack([diff[:,0]-diff[:,1],*(np.stack([pay[t+1:t+int(k)+1].sum(0) for t,k in zip(steps,h)]).T)])
                err=np.max(np.abs(expected-v[:,:3]),axis=0)
                if max(err)>.08:raise ValueError((g['id'],err))
                checks.append(dict(id=g['id'],seat=seat,roots=len(steps),maximum_cent_level_error=err.tolist()))
    out['official_crosscheck']=checks;out['labels_sha256']=sha(HERE/'labels_25.npz')
    (HERE/'label_diagnostics.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':run()
