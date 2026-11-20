"""Measure a distinct recruitment order against real replies from unchanged Master3s2."""
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
path = FAMILY/'single_worker_priority/response/run.py'
spec = importlib.util.spec_from_file_location('recruitment_verified_position', path)
src = importlib.util.module_from_spec(spec)
spec.loader.exec_module(src)
src.HERE = HERE
HORIZON = 32


def protocol():
    raw = (HERE/'search/results.json.gz').read_bytes()
    rows = json.loads(gzip.decompress(raw))['rows']
    candidates = []
    for row in rows:
        a,b = [row['equal_work'][name][0] for name in ('v4','critic')]
        if not row['same_work'] or not b['menu'] or b['menu'][-1] in a['menu']:
            continue
        ticket = row['ticket']
        depth = b['depth_by_arm'][-1]
        order = b['menu'][-1]
        if depth < 4 or order == a['issued']:
            continue
        priority = (-int(row['reliable_change'] and b['issued']==order), -depth,
                    ticket['seed'],ticket['seat'],ticket['step'])
        candidates.append((priority,row,a,order))
    if not candidates:
        return dict(schema='base-recruitment-native-replies/1',source=src.check.sha(raw),examiner=src.ORIGINAL_SHA,
                    horizon=HORIZON,ticket=None)
    _,row,parent,alternate = min(candidates,key=lambda r:r[0])
    t = row['ticket']
    return dict(schema='base-recruitment-native-replies/1',source=src.check.sha(raw),examiner=src.ORIGINAL_SHA,
                horizon=HORIZON,selection='Prefer a stable actual selection of the new menu entry; otherwise the earliest ticket among those with greatest completed depth >=4.',
                ticket=t|dict(orders={'parent':parent['issued'],'alternative':alternate},
                history_source_package=src.PARENT_SHA,own_public_sha256=row['public_sha256'],
                initial_cost=row['cost'],rival_rng=t['rng'] ^ 0x7D4A1F6B))


def execute(p, cpu):
    os.sched_setaffinity(0,{cpu})
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path[:0]=[str(ROOT/'code/env_src/carbon_challenge-master'),str(FAMILY/'symmetry_value/terminal_pairs')]
    import branches
    runner = src.load_runner()
    runner.PLAN = p
    freight = branches.freight
    live = freight.load('rent_live',freight.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    t = p['ticket']
    s = t['seat']
    modules = [freight.load(f'rent_seat_{i}',src.SOURCE if i==s else src.ORIGINAL) for i in (0,1)]
    assert src.check.sha(src.SOURCE.read_bytes())==src.PARENT_SHA
    assert src.check.sha(src.ORIGINAL.read_bytes())==src.ORIGINAL_SHA
    models, cfg, env, sim, obs, prefix = runner.replay_root(t,modules,freight,live)
    assert all(m._TEMP==1 and m._S_ABORT==.58 and m._SEED is None and m._net() is not None for m in models)
    warmed = [freight.measured(models[i],obs[i],cfg) for i in (0,1)]
    snapshot = [copy.deepcopy(freight.original_state.dynamic(m)) for m in models]
    configs = [(m._RX_CFG[0],copy.deepcopy(m._RX_CFG[1])) for m in models]
    bank = [v['remainingOverageTime'] for v in obs]
    official = freight.official_state.take(env,bank)
    sim_root = copy.deepcopy(sim)
    result = {}
    for arm, cmd in t['orders'].items():
        current_bank = freight.official_state.restore(env,official)
        sim = copy.deepcopy(sim_root)
        for m, state, conf in zip(models,snapshot,configs,strict=True):
            for key, value in copy.deepcopy(state).items():
                setattr(m,key,value)
            m._RX_CFG[:] = [conf[0],copy.deepcopy(conf[1])]
        rows = []
        accounts = [Counter(),Counter()]
        models[s]._ST['tr'].set_actions(obs[s]['players'],cmd)
        for offset in range(HORIZON):
            if env.done:
                break
            observed = [freight.public(env,i,current_bank) for i in (0,1)]
            actual = list(warmed) if offset==0 else [freight.measured(models[i],observed[i],cfg) for i in (0,1)]
            if offset==0:
                actual[s] = (cmd,actual[s][1])
            moves = [x[0] for x in actual]
            clocks = [x[1] for x in actual]
            current_bank = [b-max(0,c['wall']-1.) for b,c in zip(current_bank,clocks,strict=True)]
            cash = runner.checked_step(env,sim,moves)
            paid = freight.econ.cash_flow(sim.cash.events)
            for i in (0,1):
                accounts[i].update(paid[i])
            rows.append(dict(step=offset+t['step'], orders=moves, clocks=clocks, cash=cash,
                             events=sim.cash.events[:], players=copy.deepcopy(sim.obs(0)['players'])))
        for i in (0,1):
            assert abs(sum(accounts[i].values())-rows[-1]['cash'][i]+prefix['incoming_cash'][i])<1e-5
        result[arm]=dict(rows=rows,cash_accounts=accounts,cash_margin=rows[-1]['cash'][s]-rows[-1]['cash'][1-s])
        print(arm, len(rows), result[arm]['cash_margin'], flush=True)
    assert result['parent']['rows'][0]['orders'][1-s]==result['alternative']['rows'][0]['orders'][1-s]
    out = dict(protocol_sha256=src.check.sha(src.check.encode(p)), initial=prefix, fresh_root_enemy=warmed[1-s][0],
               outcomes=result, scoring='Official-environment wallets after a 32-turn development window, not a terminal match score.')
    (HERE/'response.json.gz').write_bytes(gzip.compress(src.check.encode(out),mtime=0))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    a=cli.parse_args()
    p=protocol()
    raw=src.check.encode(p)
    dest=HERE/'response_protocol.json'
    if a.register:
        assert not dest.exists() or dest.read_bytes()==raw
        dest.write_bytes(raw)
    else:
        assert dest.read_bytes()==raw and p['ticket'] is not None
        execute(p,a.cpu)
