"""Fit once on the 79 approved training maps, without reading validation play."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from data import HERE,OLD,STRIDE,dataset,digest,splits
from value import CashMargin30,FEATURE_NAMES,HORIZON

L2=.01


def freeze(path,data):
    if path.exists() and path.read_bytes()!=data:
        raise ValueError('existing frozen artifact disagrees: '+path.name)
    if not path.exists():
        with path.open('xb') as f:f.write(data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True,type=Path);a=p.parse_args()
    items=splits()['train']
    x,y,base,mid,seat,t=dataset(items,a.old)
    counts=np.bincount(mid)
    weights=1/(len(counts)*counts[mid])
    scale=np.sqrt(weights@(x*x))
    scale=np.where(scale>1e-8,scale,1.)
    z=x/scale
    normal=(z.T*weights)@z
    coef=np.linalg.solve(normal+L2*np.eye(len(FEATURE_NAMES)),z.T@(weights*y))
    path=HERE/'cash_margin_30.npz';buf=io.BytesIO()
    np.savez(buf,coef=coef,scale=scale,names=np.array(FEATURE_NAMES),horizon=HORIZON)
    freeze(path,buf.getvalue())
    deployed=CashMargin30(path)
    if not np.array_equal(coef,deployed.coef) or not np.array_equal(scale,deployed.scale):
        raise ValueError('checkpoint round-trip failed')
    receipt={'horizon':HORIZON,'stride':STRIDE,'rows':len(y),'map_count':len(items),'num_weights':len(coef),
             'ridge_mean_loss_lambda':L2,'weighting':'equal total weight per map, then per row within that map',
             'feature_fields':FEATURE_NAMES,'checkpoint_sha256':digest(path),'training_index_sha256':digest(OLD),
             'contract_sha256':digest(HERE/'CONTRACT.md'),'runtime_sha256':digest(HERE/'value.py'),
             'reader_sha256':digest(HERE/'data.py'),'fit_sha256':digest(HERE/'train.py'),
             'train_games':[{'id':g['id'],'seed':g['seed'],'canonical_sha256':g['canonical_sha256'],'rows':int(n)} for g,n in zip(items,counts)]}
    freeze(HERE/'frozen.json',(json.dumps(receipt,indent=2)+'\n').encode())
    print(path,len(y),len(items),len(coef),receipt['checkpoint_sha256'],flush=True)


if __name__=='__main__':main()
