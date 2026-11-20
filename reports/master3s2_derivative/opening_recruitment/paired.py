"""Change one recruit at t=0, preserve full private controls, and continue to official scoring."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parent
ROOT=D.parents[1]
LOCK="dc66bd3279f7c743328e6552d7ef65417ccf89aacc0d976cb0c0500d6e97a211"

def sha(b):return hashlib.sha256(b).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def put(path,x):
    b=(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':b=gzip.compress(b,mtime=0)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(b);tmp.replace(path);return sha(b)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int);a=parser.parse_args()
    raw=(HERE/'protocol.json').read_bytes();assert sha(raw)==LOCK;plan=json.loads(raw)
    tick=next(r for r in plan['tickets'] if r['seed']==a.seed)
    assert os.sched_getaffinity(0)=={tick['cpu']}
    hist=module('opening_tools',D/'endgame_cash_projection/response_check/recovery.py')
    F=hist.freight;live=hist.live
    archive=(D/tick['source']).read_bytes();assert sha(archive)==tick['zip_sha256']
    with zipfile.ZipFile(D/tick['source']) as z:
        reg=z.read('parent/registration.json')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);r=p/'r.json';r.write_bytes(reg);live.configure(r,p/'x',p/'l');env,cfg=live.official()
    assert 'randomSeed' not in cfg
    sim=F.econ.FastCarbon(cfg).reset(seed=tick['seed'])
    assert sha(bytes(map(int,sim.carbon)))==tick['terrain']
    sim.cash=F.econ.CashEvents(sim.cash)
    paths=[ROOT/'data/master3s2_source.py',D/'cold_start/candidate.py']
    assert [sha(path.read_bytes()) for path in paths]==[plan['original'],plan['subject']]
    mods=[F.load(f'opening_native_{i}',path) for i,path in enumerate(paths)]
    for m in mods:
        assert m._SEED is None and m._S_ABORT==.58 and m._TEMP==1 and m._SEARCH_OK
        m._torch.set_num_threads(1);m._net()
    bank=[float(live.RULE['overage_seconds_per_player'])]*2
    startobs=[F.public(env,i,bank) for i in (0,1)]
    assert [F.digest(o) for o in startobs]==tick['root_observation_hash'] and not any(startobs[0]['carbon']) and not any(startobs[1]['carbon'])
    root_calls=[F.measured(m,startobs[i],cfg) for i,m in enumerate(mods)]
    # The registered opening comes from the recorded public orders. All future choices are made anew.
    roots=tick['historical_joint_opening']
    for i,m in enumerate(mods):
        m._ST['tr'].set_actions(startobs[i]['players'],roots[i])
        m._ST['rng']=F.np.random.RandomState(tick['after_root_native_rng_seed'][i])
    dyn=[copy.deepcopy(F.original_state.dynamic(m)) for m in mods]
    fingerprints=[F.original_state.h(d) for d in dyn]
    caches=[(m._RX_CFG[0],copy.deepcopy(m._RX_CFG[1])) for m in mods]
    official=F.official_state.take(env,bank);shadow=copy.deepcopy(sim);initial=list(sim.cash)
    result={}
    for arm in tick['run_order']:
        start=time.perf_counter()
        bank=F.official_state.restore(env,official);sim=copy.deepcopy(shadow)
        for i,m in enumerate(mods):
            for k,v in copy.deepcopy(dyn[i]).items():setattr(m,k,v)
            m._RX_CFG[:]=[caches[i][0],copy.deepcopy(caches[i][1])]
            assert F.original_state.h(F.original_state.dynamic(m))==fingerprints[i]
        opening=copy.deepcopy(roots)
        if arm=='double':
            planter=[k for k,v in opening[1].items() if v=='RECPLANTER'];assert len(planter)==1
            opening[1][planter[0]]='RECCOLLECTOR'
        mods[1]._ST['tr'].set_actions(startobs[1]['players'],opening[1])
        rows=[];phase=[];running=[Counter(),Counter()]
        while not env.done:
            i=int(env.states[0].observation.step);assert i==len(rows)
            pub=[F.public(env,s,bank) for s in (0,1)]
            if i==0:
                orders=opening;clocks=[v[1] for v in root_calls]
            else:
                replies=[F.measured(m,pub[s],cfg) for s,m in enumerate(mods)]
                orders=[v[0] for v in replies];clocks=[v[1] for v in replies]
            bank=[b-max(0.,c['wall']-1.) for b,c in zip(bank,clocks)]
            sim.cash.events.clear();sim.step(*orders);env.step(orders,[])
            pp=env.states[0].observation.players;cash=[p[0] for p in pp]
            assert list(sim.cash)==cash
            flow=[dict(v) for v in F.econ.cash_flow(sim.cash.events)]
            for s in (0,1):running[s].update(flow[s])
            rows.append(dict(step=i,public_sha256=[F.digest(p) for p in pub],orders=orders,clock=clocks,cash=cash,flows=flow))
            if (i+1)%50==0 or env.done or i==0:
                for s in (0,1):assert abs(sum(running[s].values())-(cash[s]-initial[s]))<1e-5
                phase.append(dict(step=i+1,cash=cash,margin=cash[1]-cash[0],cumulative_receipts=[dict(v) for v in running],
                    workers=[dict(Counter(w[2] for w in p[2].values())) for p in pp],trees=[len(p[3]) for p in pp]))
                print(a.seed,arm,i+1,cash,flush=True)
        scores=[float(s.reward) for s in env.states]
        total=dict(official_scores=scores,seat1_margin=scores[1]-scores[0],seat1_win=scores[1]>scores[0],statuses=[s.status for s in env.states],
           steps=len(rows),phases=phase,whole_wall=time.perf_counter()-start,final_banks=bank,
           calls_per_seat=[dict(max_wall=max(r['clock'][s]['wall'] for r in rows),slow=sum(r['clock'][s]['wall']>1 for r in rows),
              total_over=sum(max(0.,r['clock'][s]['wall']-1) for r in rows),hard=sum(r['clock'][s]['wall']>3 for r in rows),
              depth_hist=dict(Counter(r['clock'][s]['H'] for r in rows))) for s in (0,1)])
        path=HERE/f'{a.seed}_{arm}.json.gz'
        result[arm]=total|{'evidence_sha256':put(path,dict(protocol=LOCK,root_public=tick['root_observation_hash'],root_original_replies=root_calls,
          actual_joint_opening=opening,postroot_private_shared_sha=fingerprints,cpu=tick['cpu'],cgroup_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
          native_clock_seconds=.58,rows=rows,result=total))}
    result['seat1_margin_change_double_minus_mixed']=result['double']['seat1_margin']-result['mixed']['seat1_margin']
    print('COMPLETE',a.seed,put(HERE/f'{a.seed}_pair.json',result),result['seat1_margin_change_double_minus_mixed'],flush=True)

if __name__=='__main__':main()
