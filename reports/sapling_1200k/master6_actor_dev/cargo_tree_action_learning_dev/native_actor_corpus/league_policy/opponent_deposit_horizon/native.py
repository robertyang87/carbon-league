"""Causal-time forecasts on all qualified original Master3s2 tapes in the frozen source inventory."""
import argparse
from collections import deque
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
from metrics import H,aggregate,assess
from predict import HERE,ROOT,SAP,SHORT,LONG,PublicDeposit,short,read

sys.path.insert(0,str(SAP))
from turning_point_ledger import CashEvents,FastCarbon

sha=lambda b:hashlib.sha256(b).hexdigest()
def require(flag,msg):
    if not flag:raise ValueError(msg)


def save(path,data):
    b=json.dumps(data,sort_keys=True,allow_nan=False,ensure_ascii=False,separators=(',',':')).encode()
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(gzip.compress(b,compresslevel=4,mtime=0) if path.suffix=='.gz' else b+b'\n')
    tmp.replace(path)


def measure(card,cfg,actions,model):
    s=card['seat'];sim=FastCarbon(cfg).reset(seed=card['seed'])
    require(sha(bytes(map(int,sim.carbon)))==card['terrain'],'physical opening differs')
    require(len(actions)==card['n'],'different action length')
    sim.cash=CashEvents(sim.cash)
    hist=deque(maxlen=11);rows=[];events=[];reference={}
    for t,joint in enumerate(actions):
        obs=sim.obs(s);hist.append(obs);curr=short.public_rows(list(hist))
        for uid,x,prior,move,group in sorted(curr,key=lambda r:str(r[0])):
            rows.append((t,uid,x))
        if curr and not reference:
            r=curr[0];uid=r[0]
            gold=short.Forecast().predict(list(hist))[uid]
            a=short.freight.FreightPredictor().predict(obs,list(hist)[:-1])[uid]
            b=short.freight.FreightPredictor(LONG/'baseline.json').predict(obs,list(hist)[:-1])[uid]
            reference=dict(t=t,uid=uid,short=[gold['bank3'],gold['bank5']],long=a['p_deposit_10'],base=b['p_deposit_10'])
        sim.cash.events.clear();sim.step(*joint)
        paid={e[3]:float(e[2]) for e in sim.cash.events if e[0]=='own_delivery' and e[1]==1-s and e[2]>0}
        events.append(paid)
    require(np.allclose(sim.cash,card['terminal_cash'],atol=.015,rtol=0.),'cash mismatches actual played game')
    if rows:
        t,uid,x=zip(*rows);x=np.stack(x);p=model.arrays(x)
        idx=next(i for i,(n,w,_) in enumerate(rows) if n==reference['t'] and w==reference['uid'])
        require(np.max(abs(p[0][idx]-[*reference['short'],reference['long']]))<2e-7
                and abs(p[1][idx,2]-reference['base'])<2e-7,'offline batch differs from frozen public entry point')
        cash=np.zeros((len(rows),3));first=np.full((len(rows),3),-1)
        for i,(step,worker,_) in enumerate(rows):
            hits=[(j+1,events[j][worker]) for j in range(step,min(step+10,len(events))) if worker in events[j]]
            for k,h in enumerate(H):
                window=[q for q in hits if q[0]<=step+h]
                if window:cash[i,k]=sum(v for _,v in window);first[i,k]=window[0][0]
        ev=assess(t,uid,x[:,0],x[:,1],p,cash,first)
    else:ev=assess([],[],[],[],[np.zeros((0,3))]*3,np.zeros((0,3)))
    return dict(seed=card['seed'],seat=s,terrain=card['terrain'],n=card['n'],
        kind=card['kind'],evaluation=ev,actual_positive_deliveries=sum(len(e) for e in events),
        actual_total_deposit=round(sum(sum(e.values()) for e in events),4),entry_parity_checked=bool(rows))


def cache_fingerprint():
    files=(HERE/'predict.py',HERE/'metrics.py',HERE/'short_baseline.json',SHORT/'head.npz',SHORT/'model.py',
           LONG/'model.json',LONG/'baseline.json',LONG/'predictor.py',ROOT/'code/fastsim.py',ROOT/'code/cnet2_feat.py',
           SAP/'turning_point_ledger.py',short.RULES/'opponent_intent.json',short.RULES/'opponent_intent.py')
    return dict(kernel=sha(inspect.getsource(measure).encode()),files={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files})


def check_cache(cache):
    marker=cache/'identity.json';current=cache_fingerprint()
    if marker.exists():
        require(read(marker)==current,'cache belongs to different models or accounting code; use a new cache path')
    else:
        require(not any(cache.glob('*.json.gz')),'unidentified earlier results in cache; use a new cache path')
        save(marker,current)


def cached(game,cfg,actions,model,cache):
    material=json.dumps([game['seed'],game['seat'],game['terrain'],cfg,actions],separators=(',',':'),sort_keys=True).encode()
    key=sha(material)
    fname=cache/(key+'.json.gz')
    if not fname.exists():save(fname,measure(game,cfg,actions,model))
    z=json.loads(gzip.decompress(fname.read_bytes()))
    for k in ['points','alerts']:z['evaluation'][k]={int(i):v for i,v in z['evaluation'][k].items()}
    return dict(z,source=game['source'],identifier=game['identifier'],kind=game['kind'],arm=game.get('arm'))


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--cache',type=Path,required=True)
    arg.add_argument('--warm-raw',action='store_true')
    a=arg.parse_args();a.cache.mkdir(parents=True,exist_ok=True);check_cache(a.cache)
    model=PublicDeposit();games=[]
    if a.warm_raw:
        # Compute only already-registered native data. The final panel is defined exclusively by sources.json.
        inventory=HERE.parents[1]/'expansion/inventory.jsonl.gz'
        src=read(SHORT/'sources.json')['native']
        used={c['terrain'] for c in src if c['split'] in ('train','calibration')}
        banned=set(read(HERE/'protocol.json')['excluded_seeds'])
        banned.update(range(620093000,620093008))
        cards=[json.loads(l) for l in gzip.open(inventory,'rt')]
        for row in cards:
            if not (row['native_outcome_supervision'] and row['native_action_supervision']) or row['duplicate_trajectory_of']:continue
            if row['physical_sha256'] in used or row['seed'] in banned:continue
            raw=json.loads(gzip.decompress((ROOT/row['path']).read_bytes()))
            g=dict(seed=row['seed'],seat=row['candidate_seat'],terrain=row['physical_sha256'],n=row['transitions'],
                   kind='raw',source=row['path'],identifier=row['trajectory_sha256'],terminal_cash=row['terminal_post_cash'])
            cached(g,raw['game_configuration'],raw['actions'][:g['n']],model,a.cache)
            if len(list(a.cache.glob('*.json.gz')))%20==0:print('cached',len(list(a.cache.glob('*.json.gz'))),flush=True)
        return
    import sources
    cards=read(HERE/'sources.json')['rows']
    for g in cards:
        cfg,actions=sources.load_game(g)
        games.append(cached(g,cfg,actions,model,a.cache))
    save(HERE/'native_games.json.gz',games)
    panels=dict(all=aggregate(games))
    for seat in (0,1):panels['observer_seat_'+str(seat)]=aggregate([g for g in games if g['seat']==seat])
    for k in sorted({g['kind'] for g in games}):panels['source_'+k]=aggregate([g for g in games if g['kind']==k])
    positive=[];false=[]
    for game in sorted(games,key=lambda g:(g['source'],g['identifier'])):
        for q in game['evaluation']['example_alarms']:
            out=positive if q['delivered']>0 else false
            if len(out)<10:out.append({k:game[k] for k in ('source','identifier','seed','seat')}|q)
    save(HERE/'native_summary.json',dict(panels=panels,early_alert_examples={'true':positive,'false':false},
         cache_fingerprint=cache_fingerprint(),files={f'{p.parent.name}/{p.name}':sha(p.read_bytes()) for p in (HERE/'sources.json',HERE/'protocol.json',
             HERE/'short_baseline.json',SHORT/'head.npz',LONG/'model.json',LONG/'baseline.json',HERE/'native_games.json.gz')}))
    print('scored',len(games),len(set(g['terrain'] for g in games)),flush=True)


if __name__=='__main__':main()
