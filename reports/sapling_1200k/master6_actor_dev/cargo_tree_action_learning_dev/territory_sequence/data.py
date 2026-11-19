"""Construct observed multi-turn territorial cash returns and the separate responding-opponent pair support."""
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP/'counterfactual_safety'))
from gather import Recorded, FastCarbon, CashEvents, D, TO, arena, read, write, sha
from collect import DIRECTIONS

RULE = read(HERE/'protocol.json')
ROOT = arena.ROOT
CACHE = Path('/workspace/sapling-all-top5-20261119/replays')
PAIRS = read(EXP/'ground_rent_native_evidence/roots.json.gz')
SIGNED = {r['case']: r for r in read(EXP.parent/'joint_response_critic_dev/signed_roots.json.gz')['rows']}
PANEL = read(EXP/'panel.json')


def terrain(seed):
    from map_identity import local_board
    return local_board(seed)


def cards():
    excluded = {terrain(s) for s in RULE['excluded_seeds']} | {r['map'] for r in PAIRS if r['split']!='train'}
    selected = []
    for c in PANEL['cards']:
        if c['terrain'] in excluded:
            continue
        selected.append(dict(path=c['path'], sha256=c['sha256'], seat=1-c['seat'], seed=c['seed'],
                             terrain=c['terrain'], split=PANEL['split'][c['terrain']], source='Master3s2', teacher='Master3s2'))
    by = collections.defaultdict(list)
    for c in read(EXP.parent/'broader_elite_backbone_dev/panel.json')['games']:
        if c['terrain_sha256'] not in excluded:
            by[c['teacher'],c['split']].append(c)
    for (teacher, split), group in sorted(by.items()):
        for c in sorted(group,key=lambda c:c['sha256'])[:8 if split=='train' else 2]:
            selected.append(dict(path=c['id']+'.json.gz', sha256=c['sha256'], seat=c['seat'], seed=c['seed'],
                                 terrain=c['terrain_sha256'], split='test' if split=='holdout' else split, source='Top5',teacher=teacher))
    assert all((c['source']=='Top5' or c['terrain']==terrain(c['seed'])) and c['seed'] not in RULE['excluded_seeds'] for c in selected)
    parts={sp:{c['terrain'] for c in selected if c['split']==sp} for sp in ['train','val','test']}
    assert not (parts['train'] & parts['val'] or parts['test'] & (parts['train']|parts['val']))
    return selected


def load_game(c):
    path=(CACHE if c['source']=='Top5' else ROOT)/c['path']
    assert sha(path)==c['sha256']
    a=read(path)
    if c['source']=='Top5':
        assert a['configuration']['randomSeed']==c['seed']
        cells=a['steps'][0][0]['observation']['full_carbon']
        assert hashlib.sha256(bytes(map(int,cells))).hexdigest()==c['terrain']
        return a['configuration'], [[x['action'] or {} for x in st] for st in a['steps'][1:]], [p[0] for p in a['steps'][-1][0]['observation']['players']],cells
    return a['game_configuration'],a['actions'],a['official_score_by_seat'],None


def credits(sim, actions, seat):
    local=np.zeros((len(actions),225),dtype=np.float64)
    delivered=np.zeros(len(actions))
    sim.cash=CashEvents(sim.cash)
    for i,orders in enumerate(actions):
        sim.cash.events.clear()
        sim.step(*orders)
        for kind,owner,amount,*fields in sim.cash.events:
            sign=1 if owner==seat else -1
            if kind=='tree_absorption':
                local[i,fields[1]]+=sign*(amount-fields[2])
            if kind in ('plant','seize'):
                local[i,fields[1]]+=sign*amount
            if kind=='own_delivery':
                delivered[i]+=sign*amount
    return local,delivered


def worker_data(model,obs,cfg,orders):
    sent,slots=model.base(obs,cfg)
    feat=model.net._FEAT
    target=feat.labels(orders)
    for slot,uid in enumerate(feat.own_uid[:10]):
        if uid not in obs['players'][obs['player']][2] or slot not in slots:
            continue
        h,move,legal,_,z=slots[slot]
        choice=int(target[slot])
        if choice<0 or not legal[choice] or not np.array_equal(sent[:slot],target[:slot]):
            continue
        loc,_,role=obs['players'][obs['player']][2][uid]
        end=int(TO[loc,DIRECTIONS.index(orders.get(uid,'STAY'))])
        yield uid,role,end,slot,choice,h,move,legal,z,int(sent[slot])


def pack_rows(c,model,arr,rows):
    cfg,actions,scores,full=load_game(c)
    fresh=lambda:FastCarbon(cfg).reset(seed=c['seed'],full_carbon=full)
    sim=fresh()
    local,delivery=credits(sim,actions,c['seat'])
    if not np.allclose(sim.cash,scores,atol=.015,rtol=0):
        # Kaggle's exceptional extinction score is distinct from the terminal physical cash.
        if c['source']!='Top5' or not any(v==-100 for v in scores):
            raise ValueError(('mismatched terminal',c['seed'],list(sim.cash),scores))
    sim=fresh();seat=c['seat'];obs=sim.obs(seat);model.base(obs,cfg)
    for t in range(1,min(193,len(actions)-23)):
        sim.step(*actions[t-1]);obs=sim.obs(seat)
        if t<64 or t%8:
            model.net._FEAT.observe(obs)
            continue
        n=len(obs['players'][seat][2]);eligible=list(worker_data(model,obs,cfg,actions[t][seat]))
        counts=collections.Counter(x[1] for x in eligible)
        for uid,role,dst,slot,a,h,m,mask,z,sent in eligible:
            area=D[dst]<=2
            prior=local[t-8:t,area].sum()/8
            values=[float(local[t:t+k,area].sum()/k-prior+.25*delivery[t:t+k].sum()/(n*k)) for k in (4,24)]
            for key,val in [('h',h),('move',m),('mask',mask),('z',z),('y',values)]:
                arr[key].append(val)
            rows.append(dict(game=c['sha256'],map=c['terrain'],split=c['split'],source=c['source'],seat=seat,t=t,role=role,
                             uid=uid,slot=slot,label=a,sent=sent,w=1/counts[role],dst=dst))


def paired(model,arr,rows):
    by=collections.defaultdict(list)
    for r in PAIRS:
        if r['split']=='train' and PANEL['split'].get(r['map'])!='train':
            continue
        if r['split']=='outside' or r['seed'] in RULE['excluded_seeds']:
            continue
        by[r['source'],r['seat']].append(r)
    for (source,seat),rs in sorted(by.items()):
        f=ROOT/source; raw=read(f);cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=raw['seed'])
        by_t=collections.defaultdict(list)
        for r in rs:
            assert sha(f)==r['source_sha256']
            by_t[r['t']].append(r)
        for t in range(max(by_t)+1):
            obs=sim.obs(seat)
            if t==0 or t in by_t:
                sent,slots=model.base(obs,cfg);feat=model.net._FEAT
                for r in by_t[t]:
                    uid=r['uid'];sl=feat.own_uid.index(uid);p=obs['players'][seat][2][uid][0]
                    s=SIGNED[r['case']];h,move,m,_,z=slots[sl];arms={};late=False
                    for arm,cmd in r['own_commands'].items():
                        a=feat.labels(cmd); match=np.array_equal(sent[:sl],a[:sl]);i=int(a[sl])
                        name=next(v for v in s['arms'].values() if v['action_sha256']==r['arms'][arm]['sequence_sha256'])
                        tape=read(ROOT/name['action_file'])
                        assert tape[0][seat]==cmd and tape[0][1-seat]==s['root_opponent']
                        if len(tape)<24:
                            late=True;break
                        fork=FastCarbon(cfg);fork.set_state(sim.get_state());loc,bank=credits(fork,tape[:24],seat)
                        # Same geographic area for both branches to avoid inventing differential pre-root value.
                        area=D[p]<=2; nw=len(obs['players'][seat][2])
                        arms[arm]=dict(action=i,prefix=bool(match and m[i]),joint=bool(np.array_equal(sent,a)),ys=[
                            float(loc[:k,area].sum()/k+.25*bank[:k].sum()/(k*nw)) for k in (4,24)],
                            official=r['arms'][arm]['official_margin'])
                    if late:
                        continue
                    for key,val in [('h',h),('move',move),('mask',m),('z',z)]:
                        arr[key].append(val)
                    rows.append(dict(case=r['case'],uid=uid,t=t,source=source,seed=r['seed'],seat=seat,map=r['map'],
                                     split=r['split'],arms=arms,cmd=r['own_commands'],slot=sl))
            else:model.net._FEAT.observe(obs)
            if t<max(by_t):sim.step(*raw['actions'][t])


def main():
    assert sha(EXP/'candidate.py')==RULE['parent']
    sel=cards();write(HERE/'sources.json',sel)
    model=Recorded(EXP/'candidate.py')
    for typ in ('pairs','sequence'):
        a=collections.defaultdict(list);rows=[]
        if typ=='sequence':
            for c in sel:
                pack_rows(c,model,a,rows)
                print(c['source'],c['seed'],len(rows),flush=True)
        else:
            paired(model,a,rows)
        np.savez_compressed(HERE/(typ+'.npz'),**{k:np.stack(v) for k,v in a.items()})
        write(HERE/(typ+'.json.gz'),rows)

if __name__=='__main__':
    main()
