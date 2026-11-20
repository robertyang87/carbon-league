"""Developer-only paired opening trials against the unchanged original."""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

ROOT = FAMILY.parents[1]
ORIGINAL = ROOT/'sources/master3s2/master3s2.py'
OPPONENT_SHA = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
PANELS = {621782452:'two_response_dev', 621758608:'nn_policy_development'}
STEPS = 100


def encode(x):
    return (json.dumps(x, ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n').encode()


def registration():
    tickets = []
    for seed, panel in PANELS.items():
        folder = FAMILY/panel/'hosted'
        settlement_raw = (folder/'settlement.json').read_bytes()
        settlement = json.loads(settlement_raw)
        for row in settlement['archives']:
            if row['seed'] != seed:
                continue
            p = folder/(str(row['id'])+'.zip')
            assert sha(p.read_bytes()) == row['sha256']
            seat = row['seat']
            member = 'parent' if panel=='two_response_dev' else 'v2_vs_original'
            with zipfile.ZipFile(p) as z:
                raw = z.read(member+'/registration.json')
            manifest = json.loads(raw)
            assert manifest['files']['candidate']['sha256'] == PARENT_SHA
            assert manifest['files']['opponent']['sha256'] == OPPONENT_SHA
            tickets.append(dict(seed=seed, seat=seat, terrain=manifest['physical_map_sha256'],
                                origin=dict(path=str(p.relative_to(ROOT)), archive=sha(p.read_bytes()),
                                            member=member, registration=sha(raw), settlement=sha(settlement_raw)),
                                initial=[int(sha(f'first-tree/1/{seed}/{seat}/{i}'.encode())[:8],16) for i in (0,1)],
                                order=('parent','candidate') if (seat+len(tickets)//2)%2==0 else ('candidate','parent')))
    assert len(tickets)==4 and len({(t['seed'],t['seat']) for t in tickets})==4
    return dict(schema='first-tree-development-opening/1', original=OPPONENT_SHA,
                packages=dict(parent=PARENT_SHA,candidate=build()), horizon=STEPS, tickets=tickets,
                population='Two already exposed development maps from the deficit review, both seats, paired identical root observation. '
                           'Share a single genuine t0 opening and seed each seat independently for later temperature-one calls. '
                           'For diagnosis only: natural clocks and both full engines, first 100 transitions only.')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def execute(p, ticket, cpu):
    os.sched_setaffinity(0,{cpu})
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import tempfile
    sys.path[:0]=[str(ROOT/'code/env_src/carbon_challenge-master'),str(FAMILY/'symmetry_value/terminal_pairs')]
    import branches
    freight=branches.freight
    live=freight.load('first_tree_official',freight.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    archive=ROOT/ticket['origin']['path']
    assert sha(archive.read_bytes())==ticket['origin']['archive']
    with zipfile.ZipFile(archive) as z:
        raw=z.read(ticket['origin']['member']+'/registration.json')
    assert sha(raw)==ticket['origin']['registration']
    assert sha(ORIGINAL.read_bytes())==p['original']
    own=ticket['seat']
    packages=dict(parent=PARENT,candidate=HERE/'candidate.py')
    games={}
    root_commands=None
    for arm in ticket['order']:
        assert sha(packages[arm].read_bytes())==p['packages'][arm]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            reg=path/'registration.json'; reg.write_bytes(raw)
            live.configure(reg,path/'unused',path/'unused.lock')
            referee,cfg=live.official()
        sim=freight.econ.FastCarbon(cfg).reset(seed=ticket['seed'])
        assert sha(bytes(map(int,sim.carbon)))==ticket['terrain']==sha(bytes(referee.states[0].observation.full_carbon))
        sim.cash=freight.econ.CashEvents(sim.cash)
        bank=[float(live.RULE['overage_seconds_per_player'])]*2
        bots=[load(f'{arm}_seat_{i}',packages[arm] if i==own else ORIGINAL) for i in (0,1)]
        torch.set_num_threads(1)
        assert all(m._net() is not None and m._TEMP==1 and m._S_ABORT==.58 and m._SEED is None for m in bots)
        rows=[]
        for step in range(STEPS):
            if referee.done:
                break
            obs=[freight.public(referee,i,bank) for i in (0,1)]
            outputs=[freight.measured(m,obs[i],cfg) for i,m in enumerate(bots)]
            clocks=[x[1] for x in outputs]
            commands=[x[0] for x in outputs]
            if step==0:
                if root_commands is None:
                    root_commands=commands
                commands=root_commands
                for i,m in enumerate(bots):
                    m._ST['tr'].set_actions(obs[i]['players'],commands[i])
                    m._ST['rng']=freight.np.random.RandomState(ticket['initial'][i])
            bank=[b-max(0,c['wall']-1) for b,c in zip(bank,clocks,strict=True)]
            sim.cash.events.clear()
            sim.step(*commands)
            referee.step(commands,[])
            observed=referee.states[0].observation
            shadow=sim.obs(0)
            assert observed.players==shadow['players'] and observed.trees==shadow['trees'] and observed.full_carbon==sim.carbon
            events=sim.cash.events[:]
            rows.append(dict(step=step, public=[freight.digest(o) for o in obs], orders=commands,
                             clocks=clocks, cash=list(sim.cash), events=events,
                             trees=[len(p[3]) for p in observed.players]))
        first=[next((r['step'] for r in rows if any(e[0]=='plant' and e[1]==i for e in r['events'])),None) for i in (0,1)]
        assert rows[0]['orders']==root_commands
        goods=[sum(1 for r in rows for e in r['events'] if e[0]=='plant' and e[1]==i) for i in (0,1)]
        ledger=[Counter(),Counter()]
        for row in rows:
            for i, flow in enumerate(freight.econ.cash_flow(row['events'])):
                ledger[i].update(flow)
        assert all(abs(sum(ledger[i].values())+80-rows[-1]['cash'][i])<1e-4 for i in (0,1))
        total=dict(first_tree=first, trees_planted=goods, cash=rows[-1]['cash'], rows=rows, ledger=ledger)
        games[arm]=total
        print(ticket['seed'],own,arm,first,goods,total['cash'],flush=True)
    assert games['parent']['rows'][0]['public']==games['candidate']['rows'][0]['public']
    name=HERE/f"{ticket['seed']}_{own}.json.gz"
    raw=gzip.compress(encode(dict(protocol=sha(encode(p)), ticket=ticket, games=games, cpu=cpu)),mtime=0)
    tmp=name.with_suffix('.partial')
    tmp.write_bytes(raw)
    tmp.replace(name)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--seed',type=int)
    cli.add_argument('--seat',type=int)
    cli.add_argument('--cpu',type=int)
    a=cli.parse_args()
    p=registration()
    dst=HERE/'protocol.json'
    raw=encode(p)
    if a.register:
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(sha(raw))
    else:
        assert dst.read_bytes()==raw
        ticket=next(x for x in p['tickets'] if (x['seed'],x['seat'])==(a.seed,a.seat))
        execute(p,ticket,a.cpu)
