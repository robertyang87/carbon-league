"""Replay sealed public maps, log same-ID receipts and proportional tax, and join only to pre-action public features."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from pydantic_core import from_json
from predictor import NAMES, public_features

HERE=Path(__file__).resolve().parent; BASE=HERE.parent; ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE.parent));sys.path.insert(0,str(ROOT/'code'))
from turning_point_ledger import FastCarbon,SITES
CACHE=Path('/workspace/sapling-master6-freight-horizon');REPLAY=Path('/workspace/sapling-all-top5-20261119/replays')

def sha(blob):return hashlib.sha256(blob).hexdigest()
def gz(path):return from_json(gzip.decompress(path.read_bytes()))

class FreightCash(list):
    def clear_step(self):
        self.deposited={};self.owner={};self.before_tax={};self.reductions={}
    def __setitem__(self,side,new):
        frame=sys._getframe(1)
        if frame.f_code!=FastCarbon.step.__code__:raise ValueError('new cash writer')
        site=SITES[frame.f_lineno];local=frame.f_locals
        if site=='base_deposit':
            w=local['w'];amount=float(new-self[side]);self.deposited[w.id]=amount;self.owner[w.id]=side
            if not np.isclose(amount,w.carbon,rtol=0,atol=1e-7):raise ValueError('unexpected smelt/quantity')
        if site=='tree_absorption':
            self.reductions=local['worker_reduction']
            for uid in self.reductions:
                if uid not in self.before_tax:self.before_tax[uid]=local['self'].workers[uid].carbon
        super().__setitem__(side,new)
    def retention(self):
        return {u:max(0.,1.-red/self.before_tax[u]) for u,red in self.reductions.items() if red>0}


def target(uid,cargo,carrier_side,root,events):
    tagged=cargo;paid=0.;gross=0.;first=None
    for event in events[root:root+10]:
        uid_receipt=event['payments'].get(uid)
        if uid_receipt is not None:
            owner,amount=uid_receipt
            if owner==carrier_side:
                gross+=amount
                if amount>0:
                    if first is None:first=event['t']
                    paid+=min(tagged,amount)
            tagged=0.
        if uid not in event['alive']:tagged=0.
        tagged*=event['ret'].get(uid,1.)
    if not 0<=paid<=min(cargo,gross)+1.e-6:raise ValueError('principal exceeds cargo or settlement')
    return [paid/cargo,int(gross>0),gross,paid,-1 if first is None else first]


def add_step(sim,actions,actual=None):
    before=set(sim.workers);sim.cash.clear_step();sim.step(*actions)
    if actual is not None:
        for s,side in enumerate(actual):
            if abs(sim.cash[s]-side[0])>.011 or set(sim.player_workers[s])!=set(side[2]):raise ValueError('official cash or roster mismatch')
            for uid,(p,c,role) in side[2].items():
                w=sim.workers[uid]
                if w.pos!=p or w.player!=s or abs(w.carbon-c)>.015 or w.is_collector!=(role=='COLLECTOR'):
                    raise ValueError('official courier mismatch')
    ret=sim.cash.retention()
    for u,r in ret.items():
        if u in sim.workers and abs(sim.workers[u].carbon-sim.cash.before_tax[u]*r)>1e-6:raise ValueError('worker tax differs from engine loss')
    return dict(t=sim.step_count,alive=set(sim.workers),ret=ret,payments={u:(sim.cash.owner[u],a) for u,a in sim.cash.deposited.items() if a>0})


def check_cfg(cfg):
    if any(cfg[k]!=v for k,v in dict(moveCost=0,smeltCost=0,size=15,episodeSteps=300).items()):raise ValueError('unsupported charged cargo conditions')


def compact_obs(raw,s):return {'step':int(raw['step']),'player':s,'players':raw['players']}


def extract(card,steps):
    path=REPLAY/(card['id']+'.json.gz');b=path.read_bytes()
    if sha(b)!=card['sha256']:raise ValueError('replay changed')
    env=from_json(gzip.decompress(b));cfg=env['configuration'];check_cfg(cfg)
    fs=env['steps'];s=card['seat'];start=fs[0][0]['observation'];sim=FastCarbon(cfg).load_observation(start);sim.cash=FreightCash(sim.cash)
    events=[];roots=[];history=[]
    for t,frame in enumerate(fs[:-1]):
        obs=compact_obs(frame[0]['observation'],s)
        if t in steps:
            roots.extend((t,uid,x) for uid,x in public_features(obs,history).items())
        history=(history+[obs])[-5:];nxt=fs[t+1];events.append(add_step(sim,[x['action'] or {} for x in nxt],nxt[0]['observation']['players']))
    x=np.asarray([item[2] for item in roots],dtype=np.float32).reshape(-1,len(NAMES)); y=np.array([target(uid,xx[0],1-s,t,events) for t,uid,xx in roots],dtype=float).reshape(-1,5)
    return dict(x=x,y=y,t=np.array([t for t,u,xx in roots],dtype=np.int16),uids=np.array([u for t,u,xx in roots]),trace_sha256=np.array(sha(b)),
                bank_events=np.array(sum(sum(o==1-s for o,a in e['payments'].values() if a>0) for e in events)),steps=np.array(len(events)))

def main():
    CACHE.mkdir(exist_ok=True);manifest=json.loads((BASE/'data_split.json').read_bytes())
    if sha((BASE/'data_split.json').read_bytes())!=json.loads((HERE/'protocol.json').read_bytes())['selection_sha256']:raise ValueError('wrong public splits')
    planned={g['id']:g['steps'] for g in gz(BASE/'data_receipts.json.gz')['games']};identities=[]
    recipe=sha(Path(__file__).read_bytes()+(HERE/'predictor.py').read_bytes()+(HERE/'protocol.json').read_bytes())
    for g in manifest['games']:
        if g['split']=='excluded':continue
        out=CACHE/(g['id']+'.npz')
        if not out.exists():
            item=extract(g,set(planned[g['id']]))
            tmp=out.with_suffix('.partial')
            with tmp.open('wb') as f:np.savez_compressed(f,**item,recipe=np.array(recipe))
            os.replace(tmp,out)
        with np.load(out) as a:
            if a['recipe'].item()!=recipe or a['trace_sha256'].item()!=g['sha256']:raise ValueError('stale cache '+g['id'])
            identities.append(dict(id=g['id'],split=g['split'],observer_seat=g['seat'],opponent=g['opponent'],gid=g['gid'],rows=len(a['x']),
                                   steps=int(a['steps']),enemy_settlements=int(a['bank_events']),data_sha256=sha(out.read_bytes()),replay_sha256=g['sha256']))
        if len(identities)%40==0:print('processed',len(identities),flush=True)
    tmp=HERE/'dataset.tmp';tmp.write_text(json.dumps(dict(recipe=recipe,features=NAMES,rows=identities),indent=1,ensure_ascii=False)+'\n');os.replace(tmp,HERE/'dataset.json')

if __name__=='__main__':main()
