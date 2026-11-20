"""Reuse the native traced search, keeping complete quantized enemy boards before fitting."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
sys.path.insert(0,str(D/'rollout_readout'))
import collect
from original_policy import original, module, free_entry, PROTOCOL, sha

P = json.loads((HERE/'protocol.json').read_bytes())


def main():
    cli=argparse.ArgumentParser();cli.add_argument('split',choices=['train','val']);cli.add_argument('cache',type=Path)
    args=cli.parse_args()
    src=original()
    assert sha(src)==P['source']
    m=module(src); net=m._net(); assert m._RNET is net and len(net.blocks)==6
    init=list(m._S_COST)
    assert collect.PLAN['depths']==P['depths'] and collect.PLAN['limit_per_depth']==P['branches']
    for card in PROTOCOL['games']:
        if card['split']!=args.split: continue
        side=card['candidate_seat']; terrain=card['terrain']; path=args.cache/(card['sha256']+'.npz')
        if path.exists(): continue
        started=perf_counter();tr=m.BeliefTracker(m._Bel,side);prev=None;rows=[];xx=[];ww=[];bb=[];calls=[]
        for n,obs,cfg,own in free_entry.public_history({**card,'expert_seat':side}):
            if n in P['steps']:
                sampled,cost=collect.capture(m,tr,prev,obs,cfg,card,init);calls.append(cost)
                for board,before,w,b,cmd,depth,idx,ourcmd in sampled:
                    t=board['step'];op=1-side;ps=board['players']
                    x=m._quant(m.encode(ps,t,op,board['carbon'],[1.]*m.N,before)).astype(np.float32)
                    h=hashlib.sha256(x.tobytes()).hexdigest()
                    key=int(1000003*(n+1)+17*side)%(2**31)+7919*depth+2
                    assert m._sample(ps,op,w,b,np.random.RandomState(key),t)==cmd
                    assert t==n+depth and set(cmd)<=set(ps[op][1])|set(ps[op][2])
                    small=np.rint(x*100).astype(np.uint8)
                    assert np.array_equal(x,small.astype(np.float32)*np.float32(.01))
                    xx.append(small);ww.append(w);bb.append(b)
                    rows.append(dict(t=t,step=n,depth=depth,branch=idx,seed=key,teacher=cmd,own=ourcmd,me=side,players=ps,hash=h))
            tr.step(obs['players'],n,obs['carbon']);tr.set_actions(obs['players'],own);prev=obs['players']
        meta=dict(source=card['sha256'],terrain=terrain,map_seed=card['seed'],rows=rows,calls=calls,wall=perf_counter()-started)
        args.cache.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,x=xx,w=ww,b=bb,meta=np.frombuffer(gzip.compress(json.dumps(meta,separators=(',',':')).encode(),mtime=0),np.uint8))
        tmp.replace(path)
        print(args.split,card['seed'],side,len(rows),round(meta['wall'],2),flush=True)


if __name__=='__main__':
    os.environ['CARBON_DEBUG']='1'
    main()
