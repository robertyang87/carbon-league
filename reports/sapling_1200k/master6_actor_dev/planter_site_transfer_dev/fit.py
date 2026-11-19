"""Train on the demonstrator's own visible seat; attribute economics only to the chosen action."""
import gzip
import io
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from model import DEST,FEATURES,MOVES,OUTCOMES,features,score
from sources import ACTOR,BASE,CACHE,HERE,ROOT,read,save,sha
from turning_point_ledger import CashEvents,FastCarbon

def replay(r):
    if r['domain']=='native':
        f=ROOT/r['source'];assert sha(f)==r['sha256']
        raw=read(f); sim=FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
        return sim,raw['actions'],raw['last_post_cash'],None
    f=CACHE/r['source']
    if f.exists():
        assert sha(f)==r['sha256'];raw=read(f)
    else:
        from hashlib import sha256
        b=subprocess.check_output(['git','cat-file','blob',r['archive_git_blob']],cwd=ROOT)
        assert sha256(b).hexdigest()==r['archive_sha256']
        with tarfile.open(fileobj=io.BytesIO(b)) as archive: piece=archive.extractfile(r['source']).read()
        assert sha256(piece).hexdigest()==r['sha256']
        raw=json.loads(gzip.decompress(piece))
    tape=raw['steps']
    initial=dict(tape[0][0]['observation'])
    initial['carbon']=initial.get('full_carbon',initial['carbon'])
    sim=FastCarbon(raw['configuration']).load_observation(initial)
    return sim,[[s.get('action') or {} for s in row] for row in tape[1:]], [p[0] for p in tape[-1][0]['observation']['players']],tape

def game(rec):
    side=rec['seat'];sim,pairs,end,elite_tape=replay(rec);cfg=sim.config
    pre={};active={};payments={};data=[]
    for t,pair in enumerate(pairs):
        obs=sim.obs(side)
        if elite_tape is not None and 100<=t<200:
            real=elite_tape[t][0]['observation']
            assert real['players']==obs['players'] and real['step']==t
            entitled=set()
            for pos,v,role in obs['players'][side][2].values():entitled.update([pos] if role=='COLLECTOR' else DEST[pos,1:])
            seat_carbon=elite_tape[t][side]['observation']['carbon']
            assert all(abs(obs['carbon'][i]-seat_carbon[i])<.01 for i in entitled)
        workers=obs['players'][side][2]
        if 100<=t<200:
            for uid,(pos,c,kind) in workers.items():
                if kind!='PLANTER':continue
                # No joint command is an input. Record labels only after all five action features exist.
                x=features(obs,uid,cfg,pre.get(uid))
                order=pair[side].get(uid) or 'NONE';a=MOVES.index(order)
                data.append(dict(t=t,uid=uid,cell=pos,dest=int(DEST[pos,a]),a=a,x=x))
        pre={uid:v[0] for uid,v in workers.items()}
        sim.cash=CashEvents(list(sim.cash));sim.step(*pair)
        trees={tr.pos:tr for tr in sim.trees.values()}
        for channel,owner,v,*detail in sim.cash.events:
            if channel in ('plant','seize'):
                uid,cell=detail
                active.pop(cell,None)
                if owner==side:
                    tr=trees[cell]
                    assert tr.player==side and (t,uid) not in payments and v<=0
                    lot=dict(t=t,uid=uid,cell=cell,tree_id=tr.id,kind=channel,cost=-v,
                             absorption=0.,collision=0.)
                    payments[t,uid]=lot
                    active[cell]=lot
            elif channel in ('tree_collision_cash','tree_absorption'):
                tid,cell=(detail[2],detail[1]) if channel=='tree_collision_cash' else (detail[0],detail[1])
                lot=active.get(cell)
                if lot and owner==side and tid==lot['tree_id']:
                    lot['absorption' if channel=='tree_absorption' else 'collision']+=v
        for cell,lot in list(active.items()):
            tr=trees.get(cell)
            if tr is None or tr.player!=side or tr.id!=lot['tree_id']:del active[cell]
    assert np.allclose(list(sim.cash),end,rtol=0,atol=.015),('terminal cash mismatch',rec['source'])
    for row in data:
        t,uid,act=row['t'],row['uid'],row['a']
        lot=payments.get((t+(act!=0),uid))
        if lot and lot['cell']==row['dest']:
            net=lot['absorption']+lot['collision']-lot['cost']
            row['target']=3 if lot['kind']=='seize' else 2 if net>1e-6 else 1
            row['lot']={**lot,'net':net}
        else:row['target']=0;row['lot']=None
    return data

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=torch.nn.Linear(len(FEATURES),32)
        self.action=torch.nn.Linear(32,1,bias=False)
        self.settlement=torch.nn.Linear(32,len(OUTCOMES))
    def forward(self,x):
        h=self.hidden(x).tanh()
        return self.action(h).squeeze(-1),self.settlement(h)

def train(rows,records,use_elite):
    ix=[i for i,row in enumerate(rows) if records[row['game']]['split']=='fit' and
        (use_elite or records[row['game']]['domain']=='native')]
    maps=Counter((records[rows[i]['game']]['domain'],records[rows[i]['game']]['terrain']) for i in ix)
    counts=Counter(d for d,g in maps)
    row_mass=[(.75 if d=='native' else .25) / (counts[d]*maps[d,g]) if use_elite else 1/(counts[d]*maps[d,g])
              for d,g in [(records[rows[i]['game']]['domain'],records[rows[i]['game']]['terrain']) for i in ix]]
    x=torch.tensor(np.stack([rows[i]['x'] for i in ix]))
    a=torch.tensor([rows[i]['a'] for i in ix]);y=torch.tensor([rows[i]['target'] for i in ix])
    w=torch.tensor(np.array(row_mass)*len(ix),dtype=torch.float32)
    torch.manual_seed(27418)
    net=Head();opt=torch.optim.Adam(net.parameters(),lr=.002)
    losses=[]
    for _ in range(32):
        total=0.
        for b in torch.randperm(len(ix)).split(256):
            action,settled=net(x[b]);idx=torch.arange(len(b))
            loss=(w[b]*(torch.nn.functional.cross_entropy(action,a[b],reduction='none')+
                  torch.nn.functional.cross_entropy(settled[idx,a[b]],y[b],reduction='none'))).mean()
            opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(b)
        losses.append(total/len(ix))
    wt={name:q.detach().numpy() for name,q in [('hidden',net.hidden.weight.T),('hidden_bias',net.hidden.bias),
        ('action',net.action.weight.T),('settlement',net.settlement.weight.T),('settlement_bias',net.settlement.bias)]}
    with torch.no_grad():
        p,s,_=score(x[:128].numpy(),wt);a_,s_=net(x[:128])
    assert max(np.max(abs(p-a_.softmax(-1).numpy())),np.max(abs(s-s_.softmax(-1).numpy())))<1e-5
    return wt,losses

def observe(rows,w):
    p,settled,n=score(np.stack([r['x'] for r in rows]),w); ix=np.arange(len(rows));a=np.array([r['a'] for r in rows])
    return [(int(p[i].argmax()),int(n[i].argmax()),p[i].tolist(),settled[i,a[i]].tolist(),
             settled[i,:,2].tolist()) for i in ix]

def main():
    torch.set_num_threads(1)
    records=read(HERE/'sources.json')['records']
    rows=[]
    for k,r in enumerate(records):
        rows.extend(dict(game=k,**v) for v in game(r))
    hdr={k:[dict(fit=sum(1 for r in rows if r['game']==i),source_sha256=rec['sha256']) for i,rec in enumerate(records)
            if rec['domain']==k] for k in ('elite','native')}
    mask=Counter((records[r['game']]['domain'],records[r['game']]['split'],r['target']) for r in rows)
    receipts={}
    scores={}
    for name,elite in [('native_only',False),('primary',True)]:
        w,loss=train(rows,records,elite)
        np.savez_compressed(HERE/(name+'.npz'),**w)
        receipts[name]={'weights_sha256':sha(HERE/(name+'.npz')),'train_loss':loss,
                        'parameters':sum(v.size for v in w.values())}
        scores[name]=observe(rows,w)
    audit=[]
    for i,r in enumerate(rows):
        out={k:v for k,v in r.items() if k!='x'}
        out['feature_sha256']=__import__('hashlib').sha256(r['x'].tobytes()).hexdigest()
        for name,v in scores.items():out[name]=v[i]
        audit.append(out)
    save(HERE/'decisions.json.gz',audit)
    save(HERE/'fit.json',dict(models=receipts,label_counts={'/'.join(map(str,k)):v for k,v in mask.items()},
         source_sha256=sha(HERE/'sources.json'),protocol_sha256=sha(HERE/'protocol.json'),train_sha256=sha(Path(__file__)),
         model_sha256=sha(HERE/'model.py'),decisions_sha256=sha(HERE/'decisions.json.gz')))
    print('rows',len(rows),'primary',receipts['primary']['weights_sha256'],flush=True)
if __name__=='__main__':main()
