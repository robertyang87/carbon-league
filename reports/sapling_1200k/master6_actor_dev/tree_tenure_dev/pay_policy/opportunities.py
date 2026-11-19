"""Chronological whole-window public replay; measure one-step changes only against committed historical enemy orders."""
from collections import defaultdict
import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
D=Path(__file__).resolve().parent; ACTOR=D.parents[1]; BASE=ACTOR.parent; ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents,FastCarbon


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pack():
    p=D/'candidate.py'; meta=read(D/'package.json')
    assert sha(p)==meta['candidate_sha256'] and sha(D/'protocol.json')==meta['protocol_sha256']
    spec=importlib.util.spec_from_file_location('whole_pay_forecast',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def one_step(sim,order,who,uid):
    branch=copy.deepcopy(sim);branch.cash=CashEvents(list(branch.cash));branch.step(*order)
    ev=branch.cash.events
    accounts=[]
    for side in (0,1):
        sums=defaultdict(float)
        for kind,p,val,*_ in ev:
            if p==side:sums[kind]+=val
        accounts.append(dict(sums))
    booked=[(k,v) for k,s,v,*args in ev if k in ('plant','seize') and s==who and args[0]==uid]
    assert len(booked)<=1
    return dict(official_visible_next_cash=list(branch.cash),signed_events_by_seat=accounts,worker_buy_cost=-booked[0][1] if booked else 0.)


def run():
    m=pack(); native=read(ACTOR/'strong_teacher_dev/proposals.json.gz'); reg=read(D/'protocol.json')
    sources={r['source']:r['source_sha256'] for r in native['rows']}; assert len(sources)==32
    games=[]
    for p in sorted(sources):
        raw=read(ROOT/p); s=raw['candidate_seat']; seed=raw['seed']; cfg=raw['game_configuration']
        assert sha(ROOT/p)==sources[p] and raw['pack_sha256']==reg['parent_sha256'] and raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1'
        sim=FastCarbon(cfg).reset(seed=seed); selected=[];checked=0; window_counts=[0,0]; actual_full_calls=0
        for t,acts in enumerate(raw['actions']):
            view=sim.obs(s)
            window=0 if 45<=t<=125 else (1 if 150<=t<=235 else None)
            if window is not None:window_counts[window]+=1
            if t==0 or (window is not None and window not in m._PAY_USED):
                old=0 if t==0 else len(m._PAY_TRACE)
                out=m.agent(view,cfg)
                actual_full_calls+=1
                if len(m._PAY_TRACE)>old:
                    trace=dict(m._PAY_TRACE[-1]); assert trace['base']==acts[s]
                    trace['actual_parent']=one_step(sim,acts,s,trace['uid'])
                    joint=acts.copy();joint[s]=out
                    trace['forced_step_comparison']=one_step(sim,joint,s,trace['uid'])
                    assert (out!=acts[s])==trace['changed']
                    if trace['changed']:
                        assert {k for k in set(out)|set(acts[s]) if out.get(k)!=acts[s].get(k)}=={trace['uid']}
                    selected.append(trace)
                else: assert out==acts[s],(seed,s,t)
                checked+=1
            elif len(m._PAY_USED)<2 and t<=235:
                m._FEAT.observe(view)
            sim.step(*acts)
        assert len(selected)<=2 and len({r['window'] for r in selected})==len(selected)
        games.append(dict(seed=seed,seat=s,terrain=raw['initial_terrain_u8_sha256'],source=p,source_sha256=sources[p],turns=len(raw['actions']),
                          full_window_turns=window_counts,actual_package_calls=actual_full_calls,selected=selected))
    assert len({r['terrain'] for r in games})==16
    selections=[t for g in games for t in g['selected']]
    changed=[t for t in selections if t['changed']]
    maps={g['seed'] for g in games if any(t['changed'] for t in g['selected'])}
    charges=sum(t['forced_step_comparison']['worker_buy_cost']>0 for t in changed)
    passed=(len(changed)>=8 and len(maps)>=4 and charges>=2)
    doc=dict(protocol_sha256=sha(D/'protocol.json'),candidate_sha256=sha(D/'candidate.py'),native_source_sha256=sha(ACTOR/'strong_teacher_dev/proposals.json.gz'),
             n_games=len(games),n_maps=16,n_windows=len(games)*2,n_reached_windows=sum(x>0 for g in games for x in g['full_window_turns']),
             selected=len(selections),changed=len(changed),changed_maps=len(maps),changed_actual_1step_payments=charges,
             changed_1step_root_expense=sum(t['forced_step_comparison']['worker_buy_cost'] for t in changed),
             opportunity_gate='PASS' if passed else 'STOP',games=games)
    (D/'opportunities.json.gz').write_bytes(gzip.compress(json.dumps(doc,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    print({k:v for k,v in doc.items() if k!='games'},flush=True)


if __name__=='__main__':run()
