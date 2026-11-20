"""Fit and independently evaluate one opponent-root policy on exact archived ladder acts."""
import argparse
import gzip
import hashlib
import json
import os
import types
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from register import HERE,BASE,sha
from runtime import opponent_context,opponent_adjust

P=json.loads((HERE/'protocol.json').read_bytes())
assert sha((HERE/'protocol.json').read_bytes())=='0ab2139b00f13269e43fd1d250fffe6cb1f88f0decfd0b62829325392e55786a'
ROLES=('collector','planter','base')

def model(source):
    m=types.ModuleType('native');exec(compile(source,'online.py','exec'),m.__dict__)
    m._torch.set_num_threads(1);m._net()
    return m

def write(name,v):
    (HERE/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def frames(r):
    path=HERE.parent/r['path'];b=path.read_bytes();assert sha(b)==r['sha'];j=json.loads(gzip.decompress(b))
    for t, pair in enumerate(j['steps'][:-1]):
        if t>=200:break
        obs={'player':r['seat'],'step':t,'players':pair[0]['observation']['players'],'carbon':pair[r['seat']]['observation']['carbon'],'trees':pair[0]['observation'].get('trees',{}),'remainingOverageTime':60}
        yield t, obs,[q['action'] or {} for q in j['steps'][t+1]],j['configuration']

def sample_game(m,r):
    s=r['seat'];e=1-s
    tr=m.BeliefTracker(m._Bel,s);prev=None;blocks=[]
    for t,o,a,_ in frames(r):
        players=o['players'];est,_=tr.step(players,t,o['carbon'])
        if t in P['steps']:
            xo=m._quant(m.encode(players,t,e,est,[1.]*225,prev))[None]
            w,b,_=m._NET(xo.astype(np.float32),m._SID)
            w,b=m._opponent_root_readout(xo,w,b,players,e)
            w=m._feedback_adjust(w,players,e,m._FEEDBACK_WEIGHTS,m._OPP_STEPS,m.FEAT_DIST)
            ctx=opponent_context(players,e,t,m._OPP_STEPS,m.FEAT_DIST)
            blocks.append((t,players,ctx,w,b,a[e]))
        tr.set_actions(players,a[s]);prev=players
    return blocks

def eligible_bases(players,enemy,truth):
    # Outcome-side mask: an earlier base in native insertion order may consume the last slot/30.
    cash,bases,workers,_=players[enemy];roster=len(workers);available=set()
    for uid in bases:
        if cash>=30 and roster<10:
            available.add(uid)
            if uid in truth and truth[uid] not in ('NONE',None):
                cash-=30;roster+=1
    return available

def observations(m,g,enemy):
    data=[[] for k in ROLES]
    for t,players,ctx,w,b,truth in g:
        eligible=eligible_bases(players,enemy,truth)
        for role,uid,pos,f,focus in ctx:
            if role==2 and uid not in eligible:continue
            z=(w if role<2 else b)[0,pos];y=(m.WIDX if role<2 else m.BIDX)[truth.get(uid,'NONE')]
            data[role].append((z,f,y))
    return data

def fit(data):
    coeff=[]
    for k in range(3):
        items=[];weights=[]
        for g in data:
            a=g[k]
            if not a:continue
            items.extend(a);weights.extend([1/len(a)]*len(a))
        w=np.array(weights);w/=w.sum();z,f,y=(np.stack(xx) for xx in zip(*items));z=z.astype(float);f=f.astype(float)
        def objective(c):
            h=np.tanh((f@c[1:])/1.5)
            pred=z*c[0]+1.5*h
            q=softmax(pred,1);val=float(w @ (logsumexp(pred,1)-pred[np.arange(len(y)),y]))
            q[np.arange(len(y)),y]-=1
            grad=np.r_[(w[:,None]*q*z).sum(),np.einsum('n,na,naf->f',w,q*(1-h*h),f)]
            reg=c-np.r_[1.,np.zeros(7)]
            return val+.04*(reg@reg),grad+.08*reg
        s=minimize(objective,np.r_[1.,np.zeros(7)],jac=True,method='L-BFGS-B',bounds=[(.78,1.12)]+[(-2.,2.)]*7,options={'maxiter':100,'ftol':1e-11})
        assert s.success
        coeff.append(s.x.astype(np.float32).tolist())
    return coeff

def score(m,card,g,c):
    e=1-card['seat'];raw={}
    totals={'roots':len(g),'n_exact':[0,0],'invalid_orders':[0,0],'full_change':0}
    for t,players,ctx,w,b,truth in g:
        eligible=eligible_bases(players,e,truth)
        wa,ba=opponent_adjust(w.copy(),b.copy(),ctx,c)
        orders=[]
        for j,(wl,bl) in enumerate(((w,b),(wa,ba))):
            seed=(1000003*(t+1)+17*(1-e))%(2**31)+7
            ans=m._sample(players,e,wl[0],bl[0],np.random.RandomState(seed),t)
            totals['n_exact'][j]+=int(ans==truth)
            wallet,bases,workers,_=players[e];newcount=sum(uid in bases for uid in ans)
            invalid=sum(uid not in workers and uid not in bases or uid in workers and ac not in m.WIDX or uid in bases and ac not in m.BIDX for uid,ac in ans.items())
            invalid+=int(newcount>0 and (newcount*30>wallet or newcount+len(workers)>10))
            assert invalid==0
            totals['invalid_orders'][j]+=invalid;orders.append(ans)
        totals['full_change']+=int(orders[0]!=orders[1])
        for role,uid,pos,f,focus in ctx:
            if role==2 and uid not in eligible:continue
            target=truth.get(uid,'NONE');y=(m.WIDX if role<2 else m.BIDX)[target]
            for grp in ['all']+(['risk'] if focus else []):
                d=raw.setdefault(f'{ROLES[role]}_{grp}',np.zeros(9))
                d[0]+=1
                for j,z in enumerate(((w if role<2 else b)[0,pos],(wa if role<2 else ba)[0,pos])):
                    p=softmax(z.astype(float));d[1+j]+=logsumexp(z.astype(float))-z[y]
                    d[3+j]+=float(orders[j].get(uid,'NONE')==target);d[5+j]+=int(np.argmax(p)==y)
                    d[7+j]+=float( np.sum(p*p)-2*p[y]+1 )
    out={k:dict(n=int(x[0]),nll=(x[1:3]/x[0]).tolist(),correct=x[3:5].astype(int).tolist(),argmax_hits=x[5:7].astype(int).tolist(),brier=(x[7:9]/x[0]).tolist()) for k,x in raw.items()}
    return {k:card[k] for k in ('match','terrain','version','opponent_id','opponent','seat','margin')}|dict(roles=out,native=totals)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','held']);parser.add_argument('--weights-sha');o=parser.parse_args()
    src=BASE.read_bytes();assert sha(src)==P['parent'];m=model(src);ds=[];cards=[r for r in P['cards'] if r['split']==o.mode]
    wpath=HERE/'weights.json'
    if o.mode=='held':assert sha(wpath.read_bytes())==o.weights_sha
    stats=[]
    if o.mode=='train':
        for r in cards:
            ds.append(observations(m,sample_game(m,r),1-r["seat"]))
        coeff=fit(ds)
        write('weights.json',dict(parent=P['parent'],protocol=sha((HERE/'protocol.json').read_bytes()),roles=ROLES,coeff=coeff,training_counts=[sum(len(a[k]) for a in ds) for k in range(3)]))
        print('frozen',sha(wpath.read_bytes()),flush=True);return
    c=np.asarray(json.loads(wpath.read_bytes())['coeff'],np.float32)
    for r in cards:stats.append(score(m,r,sample_game(m,r),c))
    write('held.json',dict(weights=sha(wpath.read_bytes()),games=stats))
    for key in ['collector_all','planter_all','base_all','collector_risk','planter_risk']:
        xx=[s['roles'][key] for s in stats if key in s['roles']]
        print(key,sum(x['n'] for x in xx),np.array([x['nll'] for x in xx]).mean(0),np.mean([np.array(x['correct'])/x['n'] for x in xx],0),flush=True)

if __name__=='__main__':main()
