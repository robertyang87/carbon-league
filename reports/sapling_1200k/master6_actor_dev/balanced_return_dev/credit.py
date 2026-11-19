"""Trace the two explicitly post-hoc extreme training alternatives versus their same-root greedy commands."""
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

from worlds import World

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents, sums_by_seat, tree_changes


def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(sim):
    return dict(step=sim.step_count,cash=list(sim.cash),
                workers={w.id:dict(seat=w.player,cell=w.pos,carried=float(w.carbon),role='collector' if w.is_collector else 'planter')
                         for w in sim.workers.values()},
                carried_by_seat=[sum(float(w.carbon) for w in sim.workers.values() if w.player==s) for s in (0,1)],
                worker_counts=[dict(Counter('collector' if w.is_collector else 'planter' for w in sim.workers.values() if w.player==s)) for s in (0,1)],
                tree_counts=[sum(t.player==s for t in sim.trees.values()) for s in (0,1)])


def advance(w,cmd,detail):
    sim=w._sim;old=list(sim.cash);sim.cash.events.clear()
    pre=snapshot(sim) if detail else None
    previous={k:(t.player,t.pos) for k,t in sim.trees.items()} if detail else None
    w.advance(cmd)
    terms=sums_by_seat(sim.cash.events)
    for i in (0,1):
        if abs(sum(terms[i].values())-sim.cash[i]+old[i])>1e-6:raise ValueError('unexplained cash write')
    row={'t':w.step-1,'ledgers':terms}
    if detail:
        row.update(before=pre,after=snapshot(sim),physical_joint_order=cmd,tree_changes=tree_changes(previous,sim),
                   collision_reports=list(sim.last_collisions))
    return row


def period(rows):
    if not rows:return None
    c=[Counter(),Counter()]
    for r in rows:
        for i in (0,1):c[i].update(r['ledgers'][i])
    return dict(start=rows[0]['t'],end=rows[-1]['t']+1,turns=len(rows),
                starting_state=rows[0]['before'],ending_state=rows[-1]['after'],total_by_physical_seat=c)


def replay_case(r,alt,binding,reg):
    state=r['state'];h=state['header'];src=h['source'];seed=h['seed'];seat=h['seat'];t=h['t'];orig=h['original_seat']
    ref=h['choice']['greedy'];record=reg['historical_maps'][str(seed)]
    path=ROOT/src['path']
    if src['sha256']!=record['seat_raw_sha256'][str(orig)] or sha(path)!=src['sha256']:
        raise ValueError('wrong source tape')
    world=World(seed,'training')
    if h['binding']!=binding or world.terrain_sha256!=h['terrain'] or src['terrain']!=h['terrain']:
        raise ValueError('root binding mismatch')
    opening=read(path)['actions'][:t]
    if len(opening)!=t:raise ValueError('truncated common prefix')
    first={}
    world._sim.cash=CashEvents(world._sim.cash);pre=[]
    for i,j in enumerate(opening):
        row=advance(world,j,i>=t-10)
        if i>=t-10:pre.append(row)
        if world.terminal:raise ValueError('historical source terminated before root')
    branches={}
    for label,key in [('parent',ref),('sampled_alternative',alt)]:
        o=r['outcomes'][key];suffix=o['actions'];off=o['official']
        if not suffix or suffix[0][seat]!=json.loads(key):raise ValueError('wrong executed root command')
        first[label]=suffix[0][1-seat]
        w=copy.deepcopy(world);w._sim.cash=CashEvents(w._sim.cash)
        beginning=list(w._sim.cash);totals=[Counter(),Counter()];early=[]
        for i,j in enumerate(suffix):
            q=advance(w,j,i<10)
            if i<10:early.append(q)
            for s in (0,1):totals[s].update(q['ledgers'][s])
        if not w.terminal or any(w.terminal[k]!=off[k] for k in ('steps','terrain_sha256','termination','extinct_by_seat','result_by_seat')):
            raise ValueError('terminal identity')
        for s in (0,1):
            if abs(w.terminal['scores'][s]-off['scores'][s])>.005001 or abs(beginning[s]+sum(totals[s].values())-w._sim.cash[s])>1e-6:
                raise ValueError('terminal score/account mismatch')
            totals[s]['terminal_rule_adjustment']=off['scores'][s]-w._sim.cash[s]
        branches[label]=dict(key=key,total_suffix_steps=len(suffix),official=off,root_to_end_cash_by_physical_seat=totals,
                             first_turns=early,forward={str(n):period(early[:n]) for n in (3,5,10)})
    if first['parent']!=first['sampled_alternative']:raise ValueError('unequal committed root opponent command')
    a,b=branches['parent'],branches['sampled_alternative']
    delta=[]
    for s in (0,1):
        aa,bb=a['root_to_end_cash_by_physical_seat'][s],b['root_to_end_cash_by_physical_seat'][s]
        delta.append({k:bb[k]-aa[k] for k in sorted(set(aa)|set(bb))})
    d=b['official']['margin_by_seat'][seat]-a['official']['margin_by_seat'][seat]
    if abs(sum(delta[seat].values())-sum(delta[1-seat].values())-d)>1e-6:
        raise ValueError('economic differential fails to close')
    return dict(seed=seed,seat=seat,original_historical_seat=orig,t=t,source=src,chosen_worker=h['choice'],
                backward={str(n):period(pre[-n:]) for n in (3,5,10)},last_ten_historical_turns=pre,
                common_precommitted_foe_root_order=first['parent'],branches=branches,delta_cash_components_by_physical_seat=delta,
                delta_official_margin=d,delta_official_u=b['official']['u_by_seat'][seat]-a['official']['u_by_seat'][seat])


def main():
    reg=read(HERE/'maps.json');bind=read(HERE/'binding.json');plan=read(HERE/'protocol.json')
    if plan['opponent']!='frozen_it50' or sha(HERE/'run.py')!=bind['run.py']:
        raise ValueError('unexpected experiment/caller')
    collected=[];skips=[];receipts={}
    for seed in reg['training']:
        for orig in (0,1):
            for t in (205,271):
                p=HERE/'train'/f'{seed}_{orig}_{t}.json.gz'
                if not p.exists():raise RuntimeError(f'32 registered receipts not complete; pending {p.name}')
                r=read(p);receipts[p.name]=sha(p)
                h=r['header'] if r.get('skipped') else r['state']['header']
                if h['binding']!=bind or (h['seed'],h['original_seat'],h['t'])!=(seed,orig,t):
                    raise ValueError('branch source identity')
                if r.get('skipped'):
                    skips.append(p.name);continue
                ref=h['choice']['greedy'];s=h['seat'];greedy=r['outcomes'][ref]['official']['margin_by_seat'][s]
                keys=set(a['key'] for a in r['state']['samples'])-{ref}
                if set(r['outcomes'])!=keys|{ref}:raise ValueError('missing or extraneous distinct sampled command')
                for key in keys:
                    gain=r['outcomes'][key]['official']['margin_by_seat'][s]-greedy
                    collected.append((gain,p.name,key))
    if len(receipts)!=32:raise ValueError('expected 32 complete or skipped training receipts')
    collected.sort()
    negative=[v for v in collected if v[0]<-1e-7];positive=[v for v in collected if v[0]>1e-7]
    report={'scope':'Post-hoc signed extremes versus greedy among distinct drawn alternatives at 32 pre-registered training roots; no selection by results in training.',
            'opponent':'Actual frozen it50 policy responded after branch. Historical tape (which may involve Master3s2) supplies only the exact pre-root prefix.',
            'selection':'Most negative and most positive official margin changes, lexicographic tape/command tie break. One root may contain both. Missing sign is reported empty.',
            'eligible_signed_counts':dict(negative=len(negative),positive=len(positive),zero=len(collected)-len(negative)-len(positive)),
            'registered_receipts':receipts,'skipped':skips,'source_sha256':sha(Path(__file__)),'cases':{}}
    for label,pool in [('negative_counterexample',negative),('positive_example',positive)]:
        if not pool:
            report['cases'][label]=None;continue
        chosen=sorted(pool,key=lambda x:((-abs(x[0])),x[1],x[2]))[0]
        d,name,key=chosen
        item=replay_case(read(HERE/'train'/name),key,bind,reg)
        if not math.isclose(item['delta_official_margin'],d,abs_tol=1e-7):raise ValueError('case ranking payload changed')
        report['cases'][label]=item
    p=HERE/'credit.json.gz'
    packed=gzip.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    if p.exists() and p.read_bytes()!=packed:raise ValueError('previous signed cases differ')
    tmp=p.with_suffix('.tmp');tmp.write_bytes(packed);tmp.replace(p)
    print('signed post-hoc credit',report['eligible_signed_counts'],sha(p))


if __name__=='__main__':main()
