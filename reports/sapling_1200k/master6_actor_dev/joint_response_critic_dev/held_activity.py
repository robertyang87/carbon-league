"""Exhaustive action menus at every decision on eight recorded held tapes; no invented outcome for any edit."""
from collections import Counter
import argparse
import gzip
import json
import time
import numpy as np
from pathlib import Path
from actions import _jrc_context,_jrc_options
from data import D,ROOT,sha,read,compact,FastCarbon,load_parent


def one(i,cache):
    reg=read(D/'source_registry.json');g=[r for r in reg['games'] if r['split']=='held'][i];p=ROOT/g['source'];r=read(p)
    assert sha(p)==g['sha256'];cfg=r['game_configuration'];sim=FastCarbon(cfg).reset(seed=g['seed']);s=g['seat']
    model=load_parent();xs=[];types=[];ptr=[0];seconds=[];valid=[]
    for t,both in enumerate(r['actions']):
        obs=sim.obs(s);start=time.monotonic();parent=model.agent(obs,cfg);assert parent==both[s]
        x,aux,emb,y=model._jrc_root;ctx=_jrc_context(obs,cfg,model._FEAT,x,aux,emb)
        opts=_jrc_options(ctx,obs,cfg,model._FEAT,x,aux,y);seconds.append(time.monotonic()-start)
        xs.extend(v[1] for v in opts);types.extend(v[2] for v in opts);ptr.append(len(xs));valid.append(len(opts));sim.step(*both)
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache/f'{i}.npz',x=np.stack(xs),role=types,ptr=ptr,sec=seconds)
    print(i,g['seed'],s,len(valid),sum(valid),round(float(np.mean(seconds)),4),flush=True)


def report(cache):
    # Live scorer has no dependence on next observations, engine simulation or sampled enemy instructions.
    env={'np':np,'agent':None}
    exec(compile((D/'serve.py').read_text(),'joint-scorer','exec'),env)
    w=dict(np.load(D/'weights.npz'));reg=read(D/'source_registry.json');games=[];tr=[]
    for i,g in enumerate([r for r in reg['games'] if r['split']=='held']):
        z=np.load(cache/f'{i}.npz');xs=z['x'];ptr=z['ptr'];types=z['role'];sc=env['_jrc_scores'](xs,w);changes=[]
        for t,(lo,hi) in enumerate(zip(ptr,ptr[1:])):
            chosen=lo+np.argmax(sc[lo:hi]);chg=bool(chosen!=lo and sc[chosen]-sc[lo]>=100)
            if chg:changes.append(dict(t=t,role=str(types[chosen]),delta=float(sc[chosen]-sc[lo]),menu_size=int(hi-lo)))
        games.append(dict(seed=g['seed'],seat=g['seat'],terrain=g['terrain'],root_count=len(ptr)-1,chosen_changes=changes,
            signed_source_sha=g['sha256'],mean_first_public_call_and_menu_seconds=float(z['sec'].mean()),max_call_and_menu_seconds=float(z['sec'].max())))
        tr+=changes
    out=dict(weights_sha256=sha(D/'weights.npz'),games=games,root_count=sum(g['root_count'] for g in games),
             changes=len(tr),by_role=dict(Counter(r['role'] for r in tr)),by_phase=dict(Counter('0-99' if r['t']<100 else ('100-199' if r['t']<200 else '200-298') for r in tr)))
    (D/'held_activity.json.gz').write_bytes(gzip.compress(compact(out),mtime=0));print({k:v for k,v in out.items() if k!='games'})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cache',type=Path);a.add_argument('--index',type=int);args=a.parse_args()
    if args.index is None: report(args.cache)
    else:one(args.index,args.cache)
