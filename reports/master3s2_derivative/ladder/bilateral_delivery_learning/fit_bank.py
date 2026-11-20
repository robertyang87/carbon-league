"""Train the same pre-registered heads on every credited base-related cash write."""
import gzip
import json
import zipfile
from collections import Counter

import numpy as np
from register import H,ROOT,read,sha,dump
from register_bank import KINDS
from source import FastCarbon,CashEvents,rows
from model import features
from fit import neural,normalize

def labels(c):
    path=ROOT/c['source']
    assert sha(path.read_bytes())==c['sha256']
    if c['domain']=='h2h':
        cfg=ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz'
        sim=FastCarbon(json.loads(gzip.decompress(cfg.read_bytes()))['game_configuration']).reset(seed=c['seed'])
        pref=c['repeat']+'/'
        with zipfile.ZipFile(path) as z:
            moves=[json.loads(l) for l in z.read(pref+'turns.jsonl').splitlines()]
            views=[json.loads(l)['observations'][0]['players'] for l in gzip.decompress(z.read(pref+'observations.jsonl.gz')).splitlines()]
        orders=([v['response']['action'] for v in m['calls']] for m in moves)
    else:
        g=json.loads(gzip.decompress(path.read_bytes()));fs=g['steps']
        sim=FastCarbon(g['configuration']).load_observation(fs[0][0]['observation'])
        orders=([s['action'] or {} for s in fr] for fr in fs[1:])
        views=[f[0]['observation']['players'] for f in fs[:-1]]
    assert sha(bytes(map(int,sim.carbon)))==c['terrain']
    sim.cash=CashEvents(sim.cash)
    sums=[];writes=Counter()
    for t,(bs,jnt) in enumerate(zip(views,orders,strict=True)):
        assert np.allclose(list(sim.cash),[s[0] for s in bs],rtol=0,atol=.0001)
        sim.cash.events.clear();sim.step(*jnt)
        scores=np.zeros((2,len(KINDS)))
        for k,s,v,*_ in sim.cash.events:
            if k in KINDS:
                assert v>=0 and s in (0,1)
                scores[s,KINDS.index(k)]+=v
                if v:writes[k]+=1
        sums.append(scores)
    return np.asarray(sums),writes

def main():
    assert sha((H/'bank_protocol.json').read_bytes())=='5a4e0473e6734644324690f4bdd4132fbac5a698446237153e443f7c7193fb3d'
    protocol=read(H/'bank_protocol.json')
    assert all(sha((H/n).read_bytes())==v for n,v in protocol['original_code'].items())
    cards=read(H/'protocol.json')['train'];counts=Counter();groups=[];raw=[];Xp=[];Xq=[];Xa=[];y0=[];y1=[]
    totals=np.zeros(3);unchanged=0
    for c in cards:
        ledger,writes=labels(c);totals+=ledger.sum((0,1));counts.update(writes)
        for r in rows([c]):
            t=r['step'];s=r['seat'];target=ledger[t:t+10].sum(0)[[s,1-s]]
            assert np.allclose(target[:,0],r['target'],rtol=0,atol=1e-7)
            a,b,d=features(r['players'],s,t,r['order']);Xp.append(a);Xq.append(b);Xa.append(d)
            y0.append(r['target']);y1.append(target.sum(1).tolist());groups.append((r['terrain'],r['match'],s))
            unchanged+=int(target[:,1:].sum()<=1e-8)
            raw.append(target.tolist())
    hist=Counter(groups);games=Counter(key[0] for key in {g[:2] for g in hist});seats=Counter(key[:2] for key in hist)
    w=np.asarray([1/(len(games)*games[k[0]]*seats[k[:2]]*hist[k]) for k in groups])
    Xp,Xq,Xa,y0,Y=map(np.asarray,(Xp,Xq,Xa,y0,y1))
    assert sha(np.c_[Xp,Xq,Xa,y0/500.].astype('<f8').tobytes())==read(H/'fit.json')['sample_hash']
    assert abs(w.sum()-1)<1e-8
    Y=Y/500.;X,params=normalize(Xp,w);ym=w@Y
    W=np.linalg.solve(X.T@(w[:,None]*X)+.0003*np.eye(2),X.T@(w[:,None]*(Y-ym)))
    heads={'cargo':dict(**params,w=W.tolist(),b=ym.tolist())};stats={}
    for name,mat in [('state',Xq),('action',np.c_[Xq,Xa])]:heads[name],stats[name]=neural(mat,Y,w)
    ws=dump(H/'bank_weights.json',heads)
    emit=dict(weights=ws,protocol=sha((H/'bank_protocol.json').read_bytes()),training_rows=len(groups),unchanged_roots=unchanged,
              changed_roots=len(groups)-unchanged,census='Each historical game once for source receipts; overlapping windows counted for changed_roots.',
              writes=dict(counts),amount_by_kind=dict(zip(KINDS,totals.tolist())),optimizer=stats,
              original_sample_sha256=read(H/'fit.json')['sample_hash'],inclusive_sample_sha256=sha(np.c_[Xp,Xq,Xa,Y].astype('<f8').tobytes()))
    print(ws,dump(H/'bank_fit.json',emit),emit['changed_roots'],emit['amount_by_kind'],flush=True)
if __name__=='__main__':main()
