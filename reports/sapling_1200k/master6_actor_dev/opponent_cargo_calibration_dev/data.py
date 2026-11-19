"""Replay contemporaneous actions in the offline ledger; call feature code with observer history only."""
from collections import defaultdict,deque
from pathlib import Path
import gzip
import hashlib
import json
import io
import subprocess
import sys
import tarfile
import numpy as np
from model import HERE,BASE,FEATURES,public_rows,intent

ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from turning_point_ledger import FastCarbon,CashEvents
CACHE=Path('/workspace/sapling-all-top5-20261119')
SOURCES=json.loads((HERE/'sources.json').read_text())


def sha(bytes_):return hashlib.sha256(bytes_).hexdigest()


def raw_ladder(g):
    path=CACHE/g['source']
    if path.is_file():packed=path.read_bytes()
    else:
        data=subprocess.check_output(['git','cat-file','blob',SOURCES['archive_git_blob']],cwd=ROOT)
        assert sha(data)==SOURCES['archive_sha256']
        with tarfile.open(fileobj=io.BytesIO(data)) as f:packed=f.extractfile(g['source']).read()
    assert sha(packed)==g['sha256']
    return json.loads(gzip.decompress(packed))


def evaluate_turn(sim,orders):
    players=sim.obs(0)['players']; groups=defaultdict(list)
    for s,pl in enumerate(players):
        for uid,(p,c,kind) in pl[2].items():
            if kind=='COLLECTOR':
                a=intent.MOVES.index(orders[s].get(uid) or 'NONE')
                groups[int(intent.DEST[p,a])].append((s,uid,c))
    sim.cash.events.clear();sim.step(*orders);dead={q[0] for q in sim.last_collisions};bank={};engaged=set();loot=defaultdict(float)
    for kind,who,money,*d in sim.cash.events:
        if kind=='own_delivery' and money>0:bank[d[0]]=bank.get(d[0],0)+money
    loss={x[0]:x for x in sim.last_collisions}
    for cell,collection in groups.items():
        if not any(u in dead for _,u,_ in collection):continue
        for s,uid,c in collection:
            if any(other!=s and cargo>0 for other,v,cargo in collection):engaged.add(uid)
        surv=[(s,u) for s,u,c in collection if u not in dead]
        if len(surv)!=1:continue
        side,uid=surv[0]
        for other,u,cargo in collection:
            if other!=side and u in loss and loss[u][4]==side:
                loot[uid]+=loss[u][2]
    return bank,engaged,loot


def game(g,gi):
    native=g['split']!='ladder';s=g['seat']
    if native:
        b=(ROOT/g['source']).read_bytes();assert sha(b)==g['sha256'];raw=json.loads(gzip.decompress(b))
        assert raw['pack_sha256']=='9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0' and raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not raw['errors']
        cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=g['seed']);acts=raw['actions'];frames=None
    else:
        raw=raw_ladder(g);cfg=raw['configuration'];frames=raw['steps'];acts=[[p['action'] or {} for p in step] for step in frames[1:]]
        sim=FastCarbon(cfg).load_observation(frames[0][0]['observation'])
    sim.cash=CashEvents(list(sim.cash));hist=deque(maxlen=11);rows=[];events=[]
    for t,joint in enumerate(acts):
        obs=sim.obs(s)
        if frames:
            v=frames[t][s]['observation'];shared=frames[t][0]['observation']
            assert shared['players']==obs['players'] and shared['step']==t
            # The archive compresses shared player data out of seat 1 and can expose a richer board in seat 0.
            # Force each public feature to read only carbon actually entitled to the observing units.
            # The exact aggregate feature code further masks all other cells.
            public_carbon=list(v['carbon'])
            visible={int(q) for p,_,role in obs['players'][s][2].values() for q in (intent.DEST[p,:1] if role=='COLLECTOR' else intent.DEST[p,1:])}
            assert all(abs(public_carbon[j]-obs['carbon'][j])<.011 for j in visible)
            obs={**obs,'carbon':[public_carbon[j] if (t==0 or j in visible) else 0 for j in range(len(public_carbon))]}
        hist.append(obs)
        # The public method has no access to the current actions or transition accounting.
        material=public_rows(list(hist));target=joint[1-s]
        for uid,x,prior,last,group in material:
            pos=obs['players'][1-s][2][uid][0]
            rows.append((gi,t,uid,pos,x,prior,last,group,intent.MOVES.index(target.get(uid) or 'NONE')))
        events.append(evaluate_turn(sim,joint))
    if native:assert np.allclose(list(sim.cash),raw['last_post_cash'])
    else:assert np.allclose(list(sim.cash),[p[0] for p in frames[-1][0]['observation']['players']])
    out=[]
    for row in rows:
        _,t,uid,*_=row;label=[];money=[];raids=[]
        for hz in (3,5):
            future=events[t:t+hz]
            bank=sum(e[0].get(uid,0) for e in future);contest=any(uid in e[1] for e in future);loot=sum(e[2].get(uid,0) for e in future)
            label.extend([bank>0,contest]);money.append(bank);raids.append(loot)
        label=[label[0],label[2],label[1],label[3]]
        out.append((*row,label,money,raids))
    return out


def build(panel):
    cards=SOURCES['native']+SOURCES['ladder'];allrows=[]
    for i,g in enumerate(cards):
        if (g['split']=='ladder')!=(panel=='ladder'):continue
        rr=game(g,i);allrows.extend(rr);print(panel,i,g['seed'],g['seat'],len(rr),flush=True)
    def col(j):return np.asarray([r[j] for r in allrows])
    np.savez_compressed(HERE/(panel+'_features.npz'),game=col(0),t=col(1),uid=col(2),position=col(3),x=col(4),prior=col(5),last=col(6),group=col(7),
                        action=col(8),events=col(9),bank=col(10),loot=col(11))


if __name__=='__main__':build(sys.argv[1])
