"""Recover actual cash changes and trace each depositing unit back through original public commands."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
ROOT=DEV.parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
econ=load('registered_cash',ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py')
def run():
    golden=json.loads(gzip.decompress((DEV/'hosted/economy.json.gz').read_bytes()))['games']
    catalog={r['archive_sha256']:(r['seed'],r['seat']) for r in golden}
    output=[]
    for path in sorted((DEV/'hosted').glob('*.zip')):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();seed,s=catalog[digest]
        for arm in ('parent','challenger'):
            with zipfile.ZipFile(path) as z:
                rec=json.loads(z.read(arm+'/receipt.json'));turns=z.read(arm+'/turns.jsonl')
                assert hashlib.sha256(turns).hexdigest()==rec['turns_sha256']
                rows=list(map(json.loads,turns.splitlines()));obs=[f['observations'][s] for f in map(json.loads,gzip.decompress(z.read(arm+'/observations.jsonl.gz')).splitlines())]
            assert len(rows)==len(obs)==299
            cfg=econ.config();dist=econ.distances(cfg["size"]);sim=econ.FastCarbon(cfg).reset(seed=seed);sim.cash=econ.CashEvents(sim.cash)
            moves=[];cash=[];payments=[];ledger=[Counter(),Counter()]
            for t,(line,public) in enumerate(zip(rows,obs)):
                assert t==public['step']==sim.step_count and sim.obs(s)['players']==public['players']
                calls=line['calls'];a=[c['response']['action'] for c in calls];cash.append(list(sim.cash));moves.append(a)
                sim.cash.events.clear();sim.step(*a)
                payments.append([dict(seat=i,uid=detail[0],arrival=t+1,amount=value) for kind,i,value,*detail in sim.cash.events if kind=='own_delivery' and value>0])
                for who,flow in enumerate(econ.cash_flow(sim.cash.events)):ledger[who].update(flow)
            cash.append(list(sim.cash));assert cash[-1]==rec['official_scores_by_seat']
            margins=[round(v[s]-v[1-s],2) for v in cash]
            first=next((t for t in range(1,300) if margins[t]<0 and max(margins[:t])>0),None)
            longfall=next((t for t in range(1,281) if max(margins[t:t+20])<0 and max(margins[:t])>0),None)
            last=next((t for t in range(299,-1,-1) if margins[t]>0),None)
            focus={}
            for dur in (3,5,10):
                worst=min(range(300-dur),key=lambda i:margins[i+dur]-margins[i])
                accounts=[]
                for items in payments[worst:worst+dur]:
                    for p in items:
                        unit=p['uid'];side=p['seat'];arrival=p['arrival']
                        hist=[]
                        for lag in (10,5,3,1):
                            i=arrival-lag
                            if i<0:continue
                            pp=obs[i]['players'];unit_state=pp[side][2].get(unit)
                            blockers=[]
                            if unit_state:
                                pos,cargo,role=unit_state
                                for uid,v in pp[1-side][2].items():
                                    if v[2]=='COLLECTOR' and v[1]<cargo and dist[v[0]][pos]<=2:
                                        blockers.append([uid,v,moves[i][1-side].get(uid,'NONE')])
                            hist.append(dict(t=i,before=unit_state,actual=moves[i][side].get(unit,'NONE'),opposite_lighter_within_two=blockers))
                        accounts.append(dict(**p,history=hist))
                focus[dur]=dict(start=worst,end=worst+dur,from_margin=margins[worst],to_margin=margins[worst+dur],payments=accounts)
            g=next(v for v in golden if (v['seed'],v['seat'],v['arm'])==(seed,s,arm))
            for k in (0,1):assert abs(ledger[k]['own_delivery']-g['by_seat'][k]['own_delivery'])<1e-5
            output.append(dict(seed=seed,seat=s,arm=arm,archive=path.name,archive_sha=digest,
                first_deficit_after_lead=first,first_20step_deficit_after_lead=longfall,last_lead=last,
                phases={t:margins[t] for t in (0,25,50,100,150,200,225,250,275,285,299)},focus=focus,
                cash_by_t=cash,real_payments_by_t=payments,whole_by_seat=g['by_seat']))
    out=HERE/'turns.json.gz';out.write_bytes(gzip.compress(json.dumps(output,separators=(',',':'),allow_nan=False).encode(),mtime=0))
if __name__=='__main__':run()
