"""Replay four fixed, native-response suffixes. Only the acting seat's root order varies."""
import argparse
import collections
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from register import HERE,D,sha

LOCK='8f2cd778db4f9614dafbc996b494185429b0d0d6bb44364e2b06f65771cac8a4'


def save(name,obj):
    path=HERE/name;raw=(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    if path.exists() and path.read_bytes()!=raw:raise ValueError(f'Existing evidence differs: {name}')
    if not path.exists():
        tmp=path.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(path)
    return sha(raw)


def replay_root(c,mods,family,live):
    arc=D/c['archive'];assert sha(arc.read_bytes())==c['archive_sha256']; stem=c['arm']+'/'
    with zipfile.ZipFile(arc) as z:
        blob=z.read(stem+'registration.json');assert sha(blob)==c['registration_sha256'];reg=json.loads(blob)
        rr=z.read(stem+'receipt.json');receipt=json.loads(rr);assert receipt['counted'] and receipt['protocol_sha256']==sha(blob)
        turns=z.read(stem+'turns.jsonl');assert sha(turns)==receipt['turns_sha256'];frames=gzip.decompress(z.read(stem+'observations.jsonl.gz'))
    s=c['seat']; assert reg['candidate_seat']==s and reg['opponent_seat']==1-s and reg['physical_map_sha256']==c['terrain']
    for side,label,pin in ((s,'candidate',c['history_source_package']),(1-s,'opponent',PLAN['examiner'])):
        assert reg['files'][label]['sha256']==receipt['initialization'][side]['response']['sha256']==pin
    assert receipt['initialization'][1-s]['response']['original']==dict(_S_ABORT=.58,_TEMP=1.,_SEED=None,_SEARCH_OK=True)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp); f=p/'registration.json';f.write_bytes(blob);live.configure(f,p/'unused',p/'unused.lock');env,cfg=live.official()
    sim=family.econ.FastCarbon(cfg).reset(seed=c['seed']);sim.cash=family.econ.CashEvents(sim.cash)
    assert sha(bytes(env.states[0].observation.full_carbon))==c['terrain']==sha(bytes(map(int,sim.carbon)))
    bank=[float(reg['platform']['overage_seconds_per_player'])]*2
    trackers=[m.BeliefTracker(m._Bel,k) for k,m in enumerate(mods)];prev=None;psums=[];acts=[];n=c['step']
    seeds=[c['rng'] if k==s else c['rival_rng'] for k in range(2)]
    opp=mods[1-s]
    for i,(frame_line,turn_line) in enumerate(zip(frames.splitlines(),turns.splitlines(),strict=True)):
        fr=json.loads(frame_line);tu=json.loads(turn_line);assert i==fr['step']==tu['step'] and not env.done
        obs=[family.public(env,k,bank) for k in (0,1)];digests=[family.digest(o) for o in obs]
        assert obs==fr['observations'] and digests==fr['public_observation_sha256']==tu['public_observation_sha256']
        if i==n:break
        psums.append(digests);orders=[j['response']['action'] for j in tu['calls']];assert all(j['status']=='ok' for j in tu['calls']);acts.append(orders)
        if i==n-1:
            opp._ST=dict(tr=copy.deepcopy(trackers[1-s]),prev=prev,rng=family.np.random.RandomState(seeds[1-s]^0x12f191ac))
            family.measured(opp,obs[1-s],cfg)
        for k in (0,1):
            trackers[k].step(obs[k]['players'],i,obs[k]['carbon']);trackers[k].set_actions(obs[k]['players'],orders[k]);live.check_action(tu['calls'][k],obs[k]);assert tu['calls'][k]['status']=='ok'
        prev=obs[0]['players'];checked_step(env,sim,orders);bank=tu['overage_remaining_seconds']
    assert len(acts)==n and digests[s]==c['own_public_sha256'];env.steps[:-1]=[None]*n;env.logs=[]
    for side,m in enumerate(mods):m._ST=dict(tr=trackers[side],prev=prev,rng=family.np.random.RandomState(seeds[side]))
    mods[s]._S_COST[:]=c['initial_cost'];sim.cash.events.clear()
    expected=dict(public_sha256_by_seat=digests,public_prefix_sha256=family.digest(psums),orders_prefix_sha256=family.digest(acts),historical_archive_receipt_sha256=sha(rr),incoming_cash=list(sim.cash),starting_overage=bank,starting_rng_by_seat=seeds,starting_cost_by_seat=[list(m._S_COST) for m in mods],restored_transitions=n)
    return (mods,cfg,env,sim,obs,expected)


def checked_step(env,sim,orders):
    sim.cash.events.clear();sim.step(*orders);env.step(orders,[]);b=env.states[0].observation
    shadow=sim.obs(0);assert b.players==shadow['players'] and b.trees==shadow['trees'] and b.full_carbon==sim.carbon
    assert list(sim.cash)==[p[0] for p in b.players];return list(sim.cash)


def go(core):
    os.sched_setaffinity(0,{core});os.environ['CARBON_DEBUG']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path[:0]=[str(D.parents[1]/'code/env_src/carbon_challenge-master'),str(D/'symmetry_value/terminal_pairs')]
    import branches
    family=branches.freight;live=family.load('rr_root_live',family.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    # The shared suffix runner already checks both balances and restored private state.
    # Also compare IDs, positions, roles, cargos, trees, and all carbon after every transition.
    branches.advance=checked_step
    out=[]
    for c in PLAN['tickets']:
        s=c['seat'];paths=[D.parents[1]/'data/master3s2_source.py']*2;paths[s]=D/'advantage_policy_update/v2_bc.py'
        assert sha(paths[s].read_bytes())==PLAN['continuation_parent'] and sha(paths[1-s].read_bytes())==PLAN['examiner']
        mods=[family.load('root_rotation_reply_s'+str(k),p) for k,p in enumerate(paths)]
        for m in mods:assert m._TEMP==1 and m._SEED is None and m._S_ABORT==.58 and m._SEARCH_OK and m._net() is not None
        state=replay_root(c,mods,family,live);_,cfg,env,sim,obs,expected=state
        now=family.original_state.h([m._ST for m in mods]);rng=family.original_state.h(mods[s]._ST['rng'])
        actual=family.measured(mods[s],obs[s],cfg);print('restored',c['seed'],s,c['step'],expected['incoming_cash'],flush=True)
        admission=dict(**c['orders'],protocol_sha256=LOCK,candidate_sha256=PLAN['source_candidate'],continuation=[sha(p.read_bytes()) for p in paths])
        one=branches.continue_pair(c,admission,c['branch_order'],state,actual,now,rng,save)
        # Cash ledger, terminal state and clocks must exist for every physical turn of both arms.
        for arm,st in one['arms'].items():
            file=HERE/f"{c['seed']}_{s}_{c['step']}_{arm}.json.gz";assert sha(file.read_bytes())==st['evidence_sha256']
            ev=json.loads(gzip.decompress(file.read_bytes()));seq=ev['rows'];assert seq[0]['orders'][s]==c['orders'][arm] and seq[0]['orders'][1-s]==ev['root_shared_original_order']
            assert seq[0]['input_sha256']==expected['public_sha256_by_seat'] and [q['t'] for q in seq]==list(range(c['step'],c['step']+len(seq)))
            assert all(st['statuses'][k]=='DONE' for k in (0,1));st['physical_seats']=[dict(own=(k==s),official=st['scores'][k],cash=st['terminal'][k]['cash'],delivery=0.,tree_net=0.) for k in (0,1)]
            for k,per in enumerate(st['physical_seats']):
                pay=st['terminal'][k]['components'];per['delivery']=pay.get('own_delivery',0.)+pay.get('opponent_worker_delivery_to_this_base',0.)
                per['tree_net']=sum(pay.get(x,0.) for x in ('ground_rent','cargo_tax','tree_collision_cash','plant','seize'))
            st['over_one_second']=[sum(q['clocks'][k]['wall']>1 for q in seq) for k in (0,1)]
            st['max_wall']=[max(q['clocks'][k]['wall'] for q in seq) for k in (0,1)]
        out.append(one)
    save('results.json',dict(protocol_sha256=LOCK,runner_sha256=sha(Path(__file__).read_bytes()),score_basis='official referee rewards (includes extinction); current matches use full horizon or natural official termination',roots=out))


if __name__=='__main__':
    raw=(HERE/'protocol.json').read_bytes();assert sha(raw)==LOCK;PLAN=json.loads(raw)
    ap=argparse.ArgumentParser();ap.add_argument('--cpu',type=int,required=True);go(ap.parse_args().cpu)
