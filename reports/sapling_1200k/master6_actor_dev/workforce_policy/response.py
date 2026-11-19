"""One frozen base order per branch. All later commands come from v1 and reactive original Master3s2."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'root_recovery_branches'))
from branches import begin, clean, public, utility
from live import MASTER_SHA, SOURCE as OPP, module
sys.path.insert(0,str(BASE))
from tree_win_gate import validate_examiner_configuration
from tree_audited_game import exam
from it50_v3_audited_seat import audit_after_call, freeze_stats
from turning_point_ledger import CashEvents, FastCarbon
BASE_ACTIONS={'NONE','RECCOLLECTOR','RECPLANTER'}
ARMS=('parent','alternative')
V1=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
V1_SHA='9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0'
MODEL=HERE/'agent.py'
SELECTION=HERE/'roots.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def persist(path,value):
    if path.exists():
        if read(path)!=value:
            raise ValueError('sealed evidence differs at '+str(path))
        return
    raw=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+'.tmp')
    temp.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)
    temp.replace(path)


def identities(root_index):
    roots=read(SELECTION)
    g=roots['roots'][root_index]
    if (len(roots['roots'])!=2 or len({(r['seed'],r['seat'],r['t']) for r in roots['roots']})!=2 or
            roots['parent_sha256']!=V1_SHA or sha(V1.read_bytes())!=V1_SHA or
            roots['candidate_sha256']!=sha(MODEL.read_bytes()) or sha(OPP.read_bytes())!=MASTER_SHA):
        raise ValueError('wrong selection or package binding')
    src=ROOT/g['source']
    native=read(src)
    arm=src.parent
    ledger=read(arm/'results.json')
    row=next(x for x in ledger['games'] if (x['seed'],x['candidate_seat'])==(g['seed'],g['seat']))
    cert=read(arm/'official_replay.json')['runs'][arm.name]
    witness=next(x for x in cert['games'] if (x['seed'],x['seat'])==(g['seed'],g['seat']))
    stamp=sha(src.read_bytes())
    assert stamp==g['source_sha256']==row['raw_sha256']==witness['transcript_sha256']
    assert row['valid'] and ledger['status']=='complete' and cert['pack_sha256']==V1_SHA==native['pack_sha256']
    assert native['initial_terrain_u8_sha256']==witness['map_sha256'] and native['official_score_by_seat']==witness['official_reward_by_seat']
    assert (native['seed'],native['candidate_seat'])==(g['seed'],g['seat'])
    validate_examiner_configuration(native['configuration'])
    t=g['t'];s=g['seat'];bid=g['base']
    a,b=clean(g['parent_joint']),clean(g['alternative_joint'])
    if (not 0<=t<len(native['actions'])-1 or clean(native['actions'][t][s])!=a or
            clean(native['actions'][t][1-s])!=clean(g['root_opponent_fixed']) or
            {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}!={bid} or
            any(z.get(bid,'NONE') not in BASE_ACTIONS for z in (a,b))):
        raise ValueError('single-base root or taped opponent submit differs')
    return roots,g,native


def legal(obs,act):
    if not isinstance(act,dict):
        raise ValueError('invalid joint')
    own=obs['players'][obs['player']]
    for uid,cmd in act.items():
        if uid in own[1] and (cmd or 'NONE') in BASE_ACTIONS:
            continue
        if uid in own[2] and (cmd or 'NONE') in {'NONE','UP','DOWN','LEFT','RIGHT'}:
            continue
        raise ValueError(f'invalid action {uid} {cmd}')


def branch(root_index,arm,out,roots,g,raw):
    seed,s,t0=g['seed'],g['seat'],g['t'];op=1-s
    env=begin(raw);conf=raw['game_configuration']
    own=module(V1,f'workforce_parent_{arm}')
    candidate=module(MODEL,f'workforce_alternative_{arm}')
    master=module(OPP,f'workforce_examiner_{arm}')
    opts={k:getattr(master,k) for k in exam.SEARCH_KEYS}
    validate_examiner_configuration(opts)
    assert master._SEARCH_OK and master._RX and master._TEMP==1 and master._SEED is None
    tracker=master.BeliefTracker(master._Bel,op)
    # All prefix decisions were already played. Replay their public observations in order.
    for t,joint in enumerate(raw['actions'][:t0+1]):
        obs=public(env,s);enemy=public(env,op)
        if env.done or (obs['step'],enemy['step'])!=(t,t): raise ValueError('broken pre-root trajectory')
        tracker.step(enemy['players'],t,enemy['carbon']);tracker.set_actions(enemy['players'],joint[op])
        # v1's only input history is Feat. The candidate may add state, so replay its real entry point.
        if t in (0,t0):
            if clean(own.agent(obs,conf))!=clean(joint[s]):raise ValueError(f'parent action identity at t{t}')
        else:
            own._FEAT.observe(obs)
        predicted=clean(candidate.agent(obs,conf))
        if t<t0:
            env.step(joint)
            continue
        if g['base'] not in obs['players'][s][1] or predicted!=clean(g['alternative_joint']):
            raise ValueError('the published candidate does not issue the selected root joint')
        root_cmd=[dict(x) for x in joint]
        root_cmd[s]=clean(g['parent_joint'] if arm=='parent' else g['alternative_joint'])
        for side,x in enumerate(root_cmd):legal(public(env,side),x)
        root_opp=sha(json.dumps(enemy,sort_keys=True,separators=(',',':')).encode())
        env.step(root_cmd)
    rng=np.random.RandomState(g['paired_rng_seed'])
    master._ST.clear();master._ST.update(tr=tracker,prev=enemy['players'],rng=rng)
    # Both arms start with the original source's initial immutable cost values and same RNG state.
    start_cost=list(master._S_COST)
    calls=[]; orig=master._search
    def counted(*args,**kwargs):
        before=freeze_stats(master._S_STATS)
        v=orig(*args,**kwargs)
        calls.append([env.states[0].observation.step,audit_after_call(before,master._S_STATS)])
        return v
    master._search=counted
    full=raw['actions'][:t0]+[root_cmd]
    start=time.monotonic()
    status_path=out/'progress.json'
    while not env.done:
        if (env.states[0].observation.step-t0)%20==1:
            tmp=status_path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'seed':seed,'seat':s,'root':t0,'arm':arm,
                                       'next_t':env.states[0].observation.step,'original_search_calls':len(calls)})+'\n')
            tmp.replace(status_path)
        obs=public(env,s);enemy=public(env,op)
        cmd=[None,None]
        cmd[s]=own.agent(obs,conf);cmd[op]=master.agent(enemy,conf)
        legal(obs,cmd[s]);legal(enemy,cmd[op])
        env.step(cmd);full.append(cmd)
    scores=[x.reward for x in env.states];statuses=[x.status for x in env.states]
    if statuses!=['DONE','DONE'] or [x[0] for x in calls]!=list(range(t0+1,len(full))):
        raise ValueError('incomplete full-original policy continuation')
    target=out/'actions.json.gz'
    persist(target,full)
    return {'seed':seed,'candidate_seat':s,'root_t':t0,'base':g['base'],'arm':arm,'v1_sha256':V1_SHA,
            'root_policy_sha256':roots['candidate_sha256'] if arm=='alternative' else V1_SHA,
            'master_sha256':MASTER_SHA,'source_tape_sha256':g['source_sha256'],'root_selection_sha256':sha(SELECTION.read_bytes()),
            'root_opponent_observation_sha256':root_opp,'shared_rng_seed':g['paired_rng_seed'],'start_cost':start_cost,
            'official_score_by_seat':scores,'official_statuses':statuses,'margin':scores[s]-scores[op],'U':utility(scores,s),
            'terminal_t':len(full),'actual_original_search_calls':calls,'search_stats':{k:master._S_STATS[k] for k in ('searched','depth','cands','rolls','changed','single','aborted')},
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'source_runtime_sha256':sha(Path(__file__).read_bytes()),
            'full_actions_sha256':sha(target.read_bytes()),'response_seconds':round(time.monotonic()-start,4)}


def audit(g,raw,arm,d,record):
    tape=d/'actions.json.gz';actions=read(tape)
    if sha(tape.read_bytes())!=record['full_actions_sha256']:raise ValueError('branch tape changed')
    t0,s,bid=g['t'],g['seat'],g['base']
    if (actions[:t0]!=raw['actions'][:t0] or
            clean(actions[t0][s])!=clean(g['parent_joint'] if arm=='parent' else g['alternative_joint']) or
            clean(actions[t0][1-s])!=clean(g['root_opponent_fixed'])):
        raise ValueError('root replay or committed prefix differs')
    env=begin(raw)
    sim=FastCarbon(raw['game_configuration']).reset(seed=g['seed'])
    sim.cash=CashEvents(sim.cash)
    original_id=sim._new_worker_id
    hires=[]
    def capture_id(side):
        uid=original_id(side)
        charge=sim.cash.events[-1]
        if charge[1]!=side or charge[0] not in ('hire_collector','hire_planter'):
            raise ValueError('worker allocation without matching cash write')
        hires.append((charge[3],uid,charge[0],-charge[2]))
        return uid
    sim._new_worker_id=capture_id
    newcomer=None;dead=None;spawn=None;trees=set()
    focal=Counter();economic=[Counter(),Counter()];span={};pre=None
    for t,pair in enumerate(actions):
        if env.done or env.states[0].observation.step!=t:raise ValueError('terminal overrun')
        sim.cash.events.clear();hires.clear()
        if t==t0:pre=list(sim.cash)
        sim.step(*pair);env.step(pair)
        if sim.obs(s)['players']!=env.states[0].observation.players:raise ValueError(f'official cash/assets differ at {t}')
        if t<t0:continue
        ev=sim.cash.events
        if t==t0:
            named=[e for e in hires if e[0]==bid]
            if len(named)>1:raise ValueError('more than one focal recruit')
            if named:
                _,newcomer,role,price=named[0];spawn={'uid':newcomer,'role':role,'cost':price}
                focal['hire_cost']=price
        for k,who,amount,*rest in ev:
            economic[who][k]+=amount
            if k.startswith('hire_'):economic[who][k+'_count']+=1
            if newcomer and who==s and k=='own_delivery' and rest[0]==newcomer:
                focal['direct_delivered']+=amount;focal['delivery_events']+=1
        if newcomer:
            # Combat cash is paid before planting/ownership changes in the same joint transition.
            for e in ev:
                if e[0]=='tree_collision_cash' and e[1]==s and e[5] in trees:focal['first_tenure_tree_collision_cash']+=e[2]
            trees={tid for tid in trees if tid in sim.trees and sim.trees[tid].player==s}
            for e in ev:
                if e[1]==s and e[0] in ('plant','seize') and e[3]==newcomer:
                    focal['tree_purchase_cost']-=e[2];focal[e[0]+'_count']+=1
                    tree=next((v for v in sim.trees.values() if v.pos==e[4] and v.player==s),None)
                    if tree:trees.add(tree.id)
            for e in ev:
                if e[0]=='tree_absorption' and e[1]==s and e[3] in trees:
                    focal['first_tenure_tree_absorption']+=e[2]
            if newcomer not in sim.workers and dead is None:dead=t+1
        age=t+1-t0
        for label,l in [('after_10',min(10,len(actions)-t0)),('after_30',min(30,len(actions)-t0)),('to_terminal',len(actions)-t0)]:
            if age!=l:continue
            for j in (0,1):
                delta=sum(amount for k,amount in economic[j].items() if not k.endswith('_count'))
                if abs(delta-(sim.cash[j]-pre[j]))>1e-6:raise ValueError('unreconciled cash')
            available=sim.workers.get(newcomer)
            got=focal['direct_delivered']+focal['first_tenure_tree_absorption']+focal['first_tenure_tree_collision_cash']
            paid=focal['hire_cost']+focal['tree_purchase_cost']
            span[label]={'joint_transitions':age,'post_t':t+1,'cash_by_seat':list(sim.cash),'delta_cash_by_seat':[sim.cash[j]-pre[j] for j in (0,1)],
                    'cash_events_by_seat':[dict(x) for x in economic],'focal':{'recruited':spawn,'alive':bool(available) if newcomer else None,
                    'disappeared_at_post_t':dead,'cargo_remaining':available.carbon if available else None,'first_tenure_trees_still_owned':sorted(trees),
                    'attributable_direct_and_first_tenure_cash_less_hire_and_purchases':got-paid,'sources':dict(focal)}}
    scores=[x.reward for x in env.states];status=[x.status for x in env.states]
    if not env.done or status!=record['official_statuses'] or scores!=record['official_score_by_seat'] or len(actions)!=record['terminal_t']:
        raise ValueError('second independent official replay disagrees')
    return {'root_command':actions[t0][s].get(bid,'NONE'),'opponent_root':actions[t0][1-s],'spawn':spawn,'root_pre_cash':pre,'windows':span,
            'replayed_official_score_by_seat':scores,'actual_search_classes':dict(Counter(e[1] for e in record['actual_original_search_calls'])),
            'prefix_and_root_verified':True,'full_action_tape_sha256':record['full_actions_sha256']}


def worker(idx,arm):
    roots,g,raw=identities(idx)
    directory=HERE/f'response_{g["seed"]}_{g["seat"]}_{g["t"]}'/arm
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=directory/'result.json'
        if target.exists():
            record=read(target)
            if record['root_selection_sha256']!=sha(SELECTION.read_bytes()) or record['arm']!=arm or record['source_tape_sha256']!=g['source_sha256']:
                raise ValueError('stale response outcome')
        else:
            if (directory/'actions.json.gz').exists():raise ValueError('orphan actions require verification before repeating this arm')
            record=branch(idx,arm,directory,roots,g,raw)
            persist(target,record)
        persist(directory/'economy.json',audit(g,raw,arm,directory,record))
        (directory/'progress.json').unlink(missing_ok=True)
        print(idx,arm,record['margin'],record['U'],flush=True)


def summarize():
    out=[]
    for idx in range(2):
        _,g,_=identities(idx);base=HERE/f'response_{g["seed"]}_{g["seat"]}_{g["t"]}'
        parent,alt=[read(base/x/'result.json') for x in ARMS]
        if (parent['root_opponent_observation_sha256']!=alt['root_opponent_observation_sha256'] or
                parent['start_cost']!=alt['start_cost'] or parent['shared_rng_seed']!=alt['shared_rng_seed']):
            raise ValueError('branch pairing differs')
        a,b=[read(base/x/'actions.json.gz') for x in ARMS];other=1-g['seat'];t0=g['t']
        first=next((t for t in range(t0+1,min(len(a),len(b))) if clean(a[t][other])!=clean(b[t][other])),None)
        out.append({'seed':g['seed'],'seat':g['seat'],'root_t':t0,'base':g['base'],'paired_delta_margin':alt['margin']-parent['margin'],
                    'paired_delta_U':alt['U']-parent['U'],'first_different_opponent_command_t':first,
                    'arms':{x:{'outcome':read(base/x/'result.json'),'economy':read(base/x/'economy.json')} for x in ARMS}})
    persist(HERE/'response_results.json',{'arms':out,'shared_opponent_action_at_root':True,'continuation':'unchanged v1 vs live full original Master3s2 for each branch',
        'paired_initialization':'Opponent belief/history replays the original public observations and submitted commands through the common root. Search cost starts from original source defaults and both response arms use the registered same initial RNG state; these elapsed-cost and RNG states are not available from the historical tape. Interactive later states can diverge.',
        'accounting':'Each physical-seat hire/deposit/tree event is actual cash. Focal attributed cash is first-tenure income and direct deposits minus the specific hire and tree charges; the paired terminal margin includes every other consequence.'})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('cmd',choices=('one','launch','summarize'));ap.add_argument('--root',type=int,choices=(0,1));ap.add_argument('--arm',choices=ARMS)
    a=ap.parse_args()
    registered=read(SELECTION)
    if len(registered['roots'])<2:
        print('Registered source roots:',len(registered['roots']),'; equivalent split-arm games started: 0',flush=True)
        return
    if a.cmd=='one':
        if a.root is None or a.arm is None:ap.error('one requires --root and --arm')
        worker(a.root,a.arm)
    elif a.cmd=='summarize':summarize()
    else:
        def job(item):
            cpu,(idx,arm)=item
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(Path(__file__).resolve()),'one','--root',str(idx),'--arm',arm],env=env,check=True)
        with (HERE/'.response.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for idx in range(2):
                    list(pool.map(job,[(cpu,(idx,arm)) for cpu,arm in zip((2,3),ARMS)]))
            summarize()

if __name__=='__main__':
    main()
