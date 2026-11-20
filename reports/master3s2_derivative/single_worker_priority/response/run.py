"""Test one changed root action against the original, with V2 playing every later turn."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from probe import FAMILY, PARENT_SHA, SOURCE, check
sys.path.insert(0, str(FAMILY/'native_critic_update/public_search/response'))
spec = importlib.util.spec_from_file_location("shared_response_loader", FAMILY/"native_critic_update/public_search/response/run.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
load_runner = shared.load_runner

ORIGINAL = FAMILY.parents[1]/'sources/master3s2/master3s2.py'
ORIGINAL_SHA = check.ORIGINAL


def plan():
    src = (HERE.parent/'results.json.gz').read_bytes()
    study = json.loads(gzip.decompress(src))
    raw = (HERE.parent/'protocol.json').read_bytes()
    assert check.sha(raw) == study['protocol_sha256']
    parent, candidate = json.loads(raw)['packages']
    assert parent == PARENT_SHA
    selected = {}
    order = lambda row: (row['ticket']['seed'],row['ticket']['seat'])
    for row in sorted(study['rows'],key=order):
        if row['reliable_change'] and row['ticket']['step']==117:
            selected.setdefault(row['ticket']['seed'],row)
    tickets = []
    for row in list(selected.values())[:2]:
        c = row['ticket']
        a = row['live']['v4'][0]['issued']
        b = row['live']['critic'][0]['issued']
        assert a != b
        tickets.append(c | dict(orders={'parent':a,'prioritized':b}, history_source_package=parent,
                                own_public_sha256=row['public_sha256'],initial_cost=row['cost'],
                                rival_rng=c['rng']^0x7D4A1F6B,branch_order=['parent','prioritized'] if not tickets else ['prioritized','parent']))
    return dict(schema='single-worker-root-response/1',source_probe_sha256=check.sha(src),
                parent=parent,candidate=candidate,examiner=ORIGINAL_SHA,
                design='After no reproducible step-247 change, test the only reproducible step-117 change as exploratory evidence. Same real original root action in both branches. One forced root, then full public V2 and original to official completion.',
                tickets=tickets)


def execute(p, core):
    os.sched_setaffinity(0,{core})
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[k]='1'
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(FAMILY.parents[1]/'code/env_src/carbon_challenge-master'),
                    str(FAMILY/'symmetry_value/terminal_pairs')]
    import branches
    runner=load_runner()
    runner.HERE,runner.PLAN=HERE,p
    freight=branches.freight
    live=freight.load('single_worker_live',freight.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    branches.advance=runner.checked_step
    assert check.sha(SOURCE.read_bytes())==PARENT_SHA and check.sha(ORIGINAL.read_bytes())==ORIGINAL_SHA
    outcomes=[]
    for c in p['tickets']:
        seat = c['seat']
        mods=[freight.load(f'physical_{s}', SOURCE if s==seat else ORIGINAL) for s in (0,1)]
        assert all(m._TEMP==1 and m._S_ABORT==.58 and m._SEED is None and m._net() is not None for m in mods)
        state=runner.replay_root(c,mods,freight,live)
        _,cfg,_,_,obs,_=state
        prefix=freight.original_state.h([m._ST for m in mods])
        rng=freight.original_state.h(mods[seat]._ST['rng'])
        first=freight.measured(mods[seat],obs[seat],cfg)
        admission=c['orders'] | dict(candidate_sha256=p['candidate'],protocol_sha256=check.sha(check.encode(p)))
        one=branches.continue_pair(c,admission,c['branch_order'],state,first,prefix,rng,runner.save)
        for tag, arm in one['arms'].items():
            path=HERE/f"{c['seed']}_{seat}_{c['step']}_{tag}.json.gz"
            raw=path.read_bytes()
            assert check.sha(raw)==arm['evidence_sha256'] and arm['statuses']==['DONE','DONE']
            series=json.loads(gzip.decompress(raw))
            assert series['rows'][0]['orders'][seat]==c['orders'][tag]
            assert series['rows'][0]['orders'][1-seat]==series['root_shared_original_order']
        outcomes.append(one)
    runner.save('results.json',dict(protocol_sha256=check.sha(check.encode(p)),roots=outcomes))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--register',action='store_true')
    cli.add_argument('--cpu',type=int)
    args=cli.parse_args()
    p=plan()
    dst=HERE/'protocol.json'
    raw=check.encode(p)
    if args.register:
        assert not dst.exists() or dst.read_bytes()==raw
        dst.write_bytes(raw)
        print(check.sha(raw),[(c['seed'],c['seat']) for c in p['tickets']])
    else:
        assert dst.read_bytes()==raw and p['tickets']
        execute(p,args.cpu)
