"""Signed true-ladder-resource parent vs original games; exact parent-state tensor and physical orders."""
from pathlib import Path
import json, gzip, hashlib, zipfile, types, sys
import numpy as np
HERE=Path(__file__).resolve().parent
D=HERE.parent
P=json.loads((HERE/'protocol.json').read_bytes())
sha=lambda raw:hashlib.sha256(raw).hexdigest()
sys.path.insert(0,str(D/'opponent_rollout_calibration'))
from study import read

def save(path,obj):
    raw=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    path.write_bytes(raw)
    return sha(raw)

def module(source,name):
    m=types.ModuleType(name);exec(compile(source,name+'.py','exec'),m.__dict__);m._torch.set_num_threads(1);m._net()
    return m

def metadata(ticket):
    archive=D/ticket['rel']
    with zipfile.ZipFile(archive) as z: r=json.loads(z.read(ticket['arm']+'/receipt.json'))
    s=ticket['seat'];a,b=r['official_scores_by_seat'];margin=(a-b)*(1 if s==0 else -1)
    assert r['counted'] and r['official_transitions']==299 and r['winner_seat']==(s if margin>0 else 1-s if margin<0 else None)
    return dict(seed=ticket['seed'],map=ticket['map'],seat=s,margin=round(margin,2),scores=[a,b],Y=float(np.clip(margin/3000,-2.5,2.5)+.5*np.sign(margin)))

def collect(m,cards):
    allrows=[]
    for c in cards:
        tape,_=read(c);s=c['seat'];g=metadata(c);tr=m.BeliefTracker(m._Bel,s);prev=None;rows=[]
        for t,(pub,acts) in enumerate(tape):
            est,known=tr.step(pub['players'],t,pub['carbon'])
            if t in P['steps']:
                x=m._quant(m.encode(pub['players'],t,s,est,known,prev)).astype(np.float32)
                hf,lw,lb=m._POLICY.forward(x,m._SID)
                sc=m._scalars(pub['players'],t,s).reshape(1,-1)
                checkw,checkb,v=m._NET(x[None],m._SID,sc)
                err=max(float(abs(checkw[0]-lw).max()),float(abs(checkb[0]-lb).max()))
                assert err<.012 and np.isfinite(v[0])
                cash,bases,workers,_=pub['players'][s];eb=set(pub['players'][1-s][1].values())
                roster=len(workers);units=[]
                for uid,pos in bases.items():
                    cmd=acts[s].get(uid,'NONE')
                    if roster>=10 or cash<30:continue
                    units.append(('base',uid,pos,m.BIDX[cmd],0.,'base'))
                    if cmd!='NONE':roster+=1;cash-=30.
                for uid,(pos,load,kind) in workers.items():
                    z=lw[pos]
                    if kind=='COLLECTOR' and m.go_home(pos,load,m._probs(z),x,set(bases.values()),eb,t,m._HOME) is not None:continue
                    units.append(('worker',uid,pos,m.WIDX[acts[s].get(uid,'NONE')],load,kind))
                for kind,uid,pos,y,cargo,family in units:
                    z=(lw if kind=='worker' else lb)[pos]
                    rows.append(dict(t=t,stage=min(t//100,2),role=kind,y=y,preV=float(v[0]),phi=hf[pos].copy(),
                                     z=z.copy(),load=cargo,family=family,uid=uid,max_abs_backend=err))
            tr.set_actions(pub['players'],acts[s]);prev=pub['players']
        allrows.append((g,rows))
    return allrows

def feature_model():
    src=(D/'cold_start/candidate.py').read_bytes(); assert sha(src)==P['parent']
    marker=b'        return lw.T, lb.T'
    assert src.count(marker)==1
    return module(src.replace(marker,b'        return hf.T, lw.T, lb.T'),'training_own_root')

def facts(games):
    return [g|dict(rows=len(rows),backend_error=max(r['max_abs_backend'] for r in rows)) for g,rows in games]
