"""Actual 75/25 guarded command sampling, official-terminal returns, and train/eval separation."""
import copy,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;BAL=ACTOR/'balanced_return_dev';PLAN=json.loads((HERE/'protocol.json').read_bytes())
sys.path.insert(0,str(BAL))
import common as arena
from common import policy
from worlds import World
PARENT=arena.IT50;OPP=arena.PARENT;BASE=arena.ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz'
def loads():
    if arena.sha(PARENT)!=PLAN['parent_sha256'] or arena.sha(OPP)!=PLAN['opponent_sha256']:raise ValueError('policy identity')
    p=policy.load(PARENT);o=policy.load(OPP)
    return p,o
def law(root,slot):
    preds=[];prefixes=[];opts=[]
    for variant in (False,True):
        y,probs=root.decode(variant,slot)
        preds.append(probs[slot]);prefixes.append(y[:slot].copy())
        opts.append({int(a):root.decode(variant,slot,int(a))[0] for a in np.flatnonzero(probs[slot]>0)})
    if not np.array_equal(prefixes[0],prefixes[1]) or np.max(abs(preds[0]-preds[1]))>1e-10:return None
    lookup={}
    for a in np.flatnonzero(preds[0]>0):
        y,_=policy.guard(root.m,root.x,opts[0][int(a)],opts[1][int(a)])
        if int(y[slot])!=a:return None
        k=policy.key(root.f.to_actions(y))
        if k in lookup.values():return None
        lookup[int(a)]=k
    return dict(prob=preds[0].tolist(),executed=lookup)
def select(root):
    y,probs=root.decode(False);s=[]
    for i,p in probs.items():
        if np.count_nonzero(p)>1:
            mask=p>0;s.append((-float(p[mask]@np.log(p[mask])),int(i)))
    for h,i in sorted(s,key=lambda z:(-z[0],z[1])):
        result=law(root,i)
        if result:return i,h,result
    return None
def rooted(m,seed,s,t):
    world,ff,prefix,origin=arena.open_state(seed,s,t)
    x=ff[s].observe(world.obs(s));return policy.Root(m,ff[s],x),world,ff,prefix,origin
def row_features(root,slot):
    feature=sys.modules['cnet2_feat']; x=root.x;ref,_=root.decode(False)
    seq=feature.Prefix(x,root.aux)
    uf=np.zeros((feature.OWN,feature.UF),np.float32);mf=np.zeros((feature.OWN,5,feature.MF),np.float32);legal=np.zeros((feature.OWN,5),bool)
    for s,y in enumerate(ref):
        if y<0:continue
        uf[s],mf[s],legal[s]=seq.feats(s);seq.advance(s,int(y))
    return {**x,'uf':uf,'mf':mf,'legal':legal,'y':ref,'teacher':2,'value':0.,'carbon':np.zeros(225,np.int8)}
def gather():
    m,o=loads();out=[];beg=time.monotonic()
    for seed in PLAN['train_seeds']:
        for seat in (0,1):
            for t in PLAN['train_steps']:
                root,w,ff,hist,origin=rooted(m,seed,seat,t);selected=select(root)
                if not selected:
                    out.append(dict(seed=seed,seat=seat,t=t,skip=True));continue
                k,ent,dist=selected;act=dist['executed'];old=dist['prob']
                rng=np.random.default_rng(np.random.SeedSequence([192066,seed,seat,t]))
                draws=rng.choice(5,PLAN['samples_per_root'],p=old)
                greedy=policy.key(ff[seat].to_actions(m._mixed_actions(root.x,root.aux)))
                nets=[m,o] if seat==0 else [o,m]; enemy=arena.decision(o,ff[1-seat],w.obs(1-seat))
                outcomes={};trace={}
                for cmd in sorted({act[int(a)] for a in draws}|{greedy}):
                    joint=[json.loads(cmd),enemy] if seat==0 else [enemy,json.loads(cmd)]
                    terminal=arena.finish(seed,hist,w,ff,nets,first=joint)
                    v=terminal['official'];trace[cmd]=terminal['actions']
                    outcomes[cmd]=dict(U=v['u_by_seat'][seat],margin=v['margin_by_seat'][seat],score=v['scores'],steps=v['steps'])
                stem=f'{seed}_{seat}_{t}';arena.write(HERE/'training'/f'{stem}.json.gz',dict(trace=trace,source=origin))
                out.append(dict(seed=seed,seat=seat,t=t,slot=k,entropy=ent,draws=[int(a) for a in draws],old_prob=old,
                                executed=act,logp=[math.log(old[a]) for a in draws],outcomes=outcomes,parent_greedy=greedy,source=origin))
                print('trained root',stem,'branches',len(outcomes),'seconds',round(time.monotonic()-beg,1),flush=True)
    arena.write(HERE/'sampling.json',out)
if __name__=='__main__':gather()
