"""Independently replay every recorded physical command and settle at the first differing order."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

HERE=Path(__file__).resolve().parent
SAP=next(p for p in HERE.parents if (p/'turning_point_ledger.py').is_file())
ROOT=SAP.parents[1]
ENGINE=ROOT/'code/env_src/carbon_challenge-master'
sys.path[:0]=[str(SAP),str(ENGINE)]
from zerosum_env import make
from turning_point_ledger import CashEvents, FastCarbon

START,STOP=200,208
INCOME=('own_delivery','opponent_worker_delivery_to_this_base','base_collision_cash','tree_collision_cash',
        'ground_rent','cargo_tax','hire_collector','hire_planter','plant','seize','rounding')


def check(ok,msg):
    if not ok: raise ValueError(msg)


def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(doc):
    return json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def path_hash(path):
    return sha(path.read_bytes())


def clock(rows,first):
    check([r['step'] for r in rows]==list(range(first,first+len(rows))),'incomplete native clock')
    walls=[r['public_wall_seconds'] for r in rows]
    return dict(turns=len(rows), candidate_counts=Counter(int(r['candidates']) for r in rows),
                completed_depths=Counter(int(r['max_depth']) for r in rows),
                kinds=Counter(r['kind'] for r in rows), native_aborted=sum(r['native_aborted'] for r in rows),
                zero_depth_multicandidate=sum(r['candidates']>1 and r['max_depth']==0 for r in rows),
                wall_total_s=sum(walls), cpu_total_s=sum(r['process_cpu_seconds'] for r in rows),
                wall_max_s=max(walls,default=None),
                wall_p95_s=float(np.percentile(walls,95)) if walls else None)


def replay(raw,commands,root):
    """Official state is checked against independently reconstructed cash at every transition."""
    seed=raw['seed'];seat=raw['candidate_seat'];cfg={**raw['game_configuration'],'randomSeed':seed}
    np.random.seed(seed)
    env=make('carbon',configuration=cfg)
    sim=FastCarbon(cfg).reset(seed=seed)
    opening=env.states[0].observation.full_carbon
    check(list(opening)==raw['initial_full_carbon']==list(sim.carbon)
          and sha(bytes(opening))==raw['initial_terrain_u8_sha256'],'opening mismatch')
    sim.cash=CashEvents(sim.cash)
    accounts={label:[Counter(),Counter()] for label in (
        'before_first_difference','from_first_difference','registered_eight_step_window','full_game')}
    balances=[list(sim.cash)]
    for t,joint in enumerate(commands):
        check(not env.done and env.states[0].observation.step==t,'extra or missing official transition')
        sim.cash.events.clear();sim.step(*joint);env.step(joint)
        players=env.states[0].observation.players; cash=[v[0] for v in players]
        check(np.allclose(cash,sim.cash,rtol=0,atol=.011),f'wallet mismatch at {t}')
        balances.append(list(cash))
        labels=['full_game','before_first_difference' if root is None or t<root else 'from_first_difference']
        if START<=t<STOP: labels.append('registered_eight_step_window')
        for key,side,val,*meta in sim.cash.events:
            amounts={'cargo_tax':meta[2],'ground_rent':val-meta[2]} if key=='tree_absorption' else {key:val}
            check(set(amounts)<=set(INCOME),'unrecognized cash event')
            for label in labels:accounts[label][side].update(amounts)
    statuses=[v.status for v in env.states]
    rewards=[v.reward for v in env.states]
    check(env.done and statuses==['DONE','DONE'] and all(isinstance(v,(int,float)) for v in rewards),'incomplete official scoring')
    n=len(commands);boundary=n if root is None else root
    intervals={'full_game':(0,n),'before_first_difference':(0,boundary),'from_first_difference':(boundary,n),
               'registered_eight_step_window':(min(n,START),min(n,STOP))}
    for label,(lo,hi) in intervals.items():
        totals=accounts[label]
        for side in (0,1):
            check(abs(sum(totals[side].values())-(balances[hi][side]-balances[lo][side]))<=.011,
                  f'cash attribution does not close {label},seat {side}')
        accounts[label]=dict(turns_half_open=[lo,hi],opening_wallet=balances[lo],closing_wallet=balances[hi],
                             by_seat=[{k:round(v.get(k,0.),6) for k in INCOME} for v in totals])
    return dict(actions_sha256=sha(canonical(commands)),official_transitions=n,official_status=statuses,
                official_score_physical=rewards,learner_seat=seat,ending_wallet=balances[-1],
                official_margin=round(rewards[seat]-rewards[1-seat],2),economic_accounts=accounts)


def net_accounts(p1,p0,s):
    result={}
    for name in p0['economic_accounts']:
        t=p1['economic_accounts'][name]['by_seat'];p=p0['economic_accounts'][name]['by_seat']
        deltas=[{k:round(t[i][k]-p[i][k],6) for k in INCOME} for i in (0,1)]
        result[name]=dict(change_by_physical_seat=deltas,learner_minus_examiner_change={
            k:round(deltas[s][k]-deltas[1-s][k],6) for k in INCOME})
    return result


def settle(plan,out,case,stream):
    folder=out/f"{case['seed']}_{case['seat']}_stream{stream}"
    paths={name:folder/f'{name}.json.gz' for name in ('shared','parent','teacher')}
    paths['pair']=folder/'pair.json';paths['historical']=ROOT/case['path']
    digests={k:path_hash(p) for k,p in paths.items()}
    raw=read(paths['historical']);b=read(paths['shared']);pair=read(paths['pair'])
    s=case['seat'];t0=b['root_t']
    check(digests['historical']==case['sha256'] and (raw['seed'],raw['candidate_seat'])==(case['seed'],s)
          and raw['initial_terrain_u8_sha256']==case['terrain'],'archive binding')
    check(pair['identity']==b['identity'] and pair['identity']['plan']==sha(canonical(plan))
          and pair['identity']['source']==case['sha256'] and pair['identity']['stream']==stream,
          'shared identity mismatch')
    check(START<=t0<STOP and len(b['prefix'])==t0 and b['prefix'][:START]==raw['actions'][:START],
          'shared history disobeys registration')
    for t in range(START,t0):
        q=b['trace'][t]
        check(q['t']==t and q['parent']==q['teacher']==b['prefix'][t][s] and q['enemy']==b['prefix'][t][1-s],
              'changed command preceded registered fork')
    root=b['root']
    positive_treatment=root['parent']!=root['teacher']
    check(root['t']==t0 and (positive_treatment or t0==STOP-1),'improper intervention root')
    arms={};clock_tails={}
    for arm in ('parent','teacher'):
        r=read(paths[arm]);tape=r['tape']
        check(digests[arm]==pair[f'{arm}_file_sha'] and r['arm']==arm and r['shared_sha']==sha(canonical(b))
              and sha(canonical(tape))==r['tape_sha'] and r['root_t']==t0,'bad branch binding')
        check(tape[0][s]==root[arm] and tape[0][1-s]==root['enemy'],'root enemy was not simultaneous')
        settled=replay(raw,b['prefix']+tape,t0 if positive_treatment else None)
        check(settled['official_status']==r['status'] and settled['official_score_physical']==r['scores']
              and settled['ending_wallet']==r['cash'] and abs(settled['official_margin']-r['margin'])<.0001,
              'independent official result disagrees with branch')
        # A zero-dose placebo is scored as played; its two post-fork paths can still
        # differ through independent later timing of the wall-clock search.
        arms[arm]=settled
        clock_tails[arm]=clock(r['clock']['calls'],t0+1)
        check(clock_tails[arm]['turns']==len(tape)-1,'missing native continuation calls')
    shared_clock=clock(b['clock']['calls'],0);check(shared_clock['turns']==t0+1,'common root clock missing')
    observed=round(arms['teacher']['official_margin']-arms['parent']['official_margin'],2)
    check(abs(observed-pair['delta_margin'])<.0001,'paired official margin mismatch')
    return dict(seed=case['seed'],seat=s,stream=stream,terrain=case['terrain'],source_sha256=digests,
                first_different_learner_order_step=t0 if positive_treatment else None,
                common_comparison_step=t0,root_role_changes=root['roles'],
                teacher_and_parent_identical_through_end_of_window=not positive_treatment,
                root_enemy_order_sha256=sha(canonical(root['enemy'])),native_rng_after_own_first_step=b['native_first_search_rng_state_sha'],
                clock_common_including_root=shared_clock,clock_after_common_root=clock_tails,
                registered_official_margin_delta=observed,causal_credit_for_zero_treatment=0 if not positive_treatment else None,
                economic_delta=net_accounts(arms['teacher'],arms['parent'],s),arms=arms)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--destination',type=Path,default=HERE/'settled_windows.json')
    args=parser.parse_args();p=HERE/'protocol.json';plan=read(p)
    check(path_hash(HERE/'run.py')==plan['runner_sha256'],'changed original runner')
    pairs=[settle(plan,args.out,c,j) for c in plan['inputs'] for j in (0,1)]
    for c in plan['inputs']:
        group=[p for p in pairs if p['seed']==c['seed']]
        check(len({p['native_rng_after_own_first_step'] for p in group})==2,'duplicate examiner stream')
    doc=dict(protocol_sha256=path_hash(p),settler_sha256=path_hash(Path(__file__)),
             cash_simulator_sha256=path_hash(ROOT/'code/fastsim.py'),cash_accounting_sha256=path_hash(SAP/'turning_point_ledger.py'),
             official_engine_sha256={f:path_hash(ENGINE/f) for f in ('zerosum_env/core.py','zerosum_env/envs/carbon/carbon.py',
                                         'zerosum_env/envs/carbon/helpers.py','zerosum_env/envs/carbon/carbon.json')},
             pairs=pairs,physical_arms_checked=len(pairs)*2,
             scope='Post-hoc bookkeeping on prespecified historical conditional continuations. The originally measured win/score and search clocks are preserved. The shared clock includes the root once. Absolute accounts use full-game turns up to and from the first real intervention. If no treatment occurred, all outcomes are retained as a placebo; divergent clock-dependent continuations must not be credited to the teacher.')
    payload=(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if args.destination.exists():check(args.destination.read_bytes()==payload,'existing settlement changed')
    else:
        tmp=args.destination.with_name('.'+args.destination.name+'.tmp');tmp.write_bytes(payload);tmp.replace(args.destination)
    print('officially settled',doc['physical_arms_checked'],'arms:',sha(payload))


if __name__=='__main__': main()
