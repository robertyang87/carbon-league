"""Acquire fixed two-action terminal preferences on verified V2 vs original traces."""
import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
PARENT = FAMILY / 'ladder/prediction_feedback/package/candidate.py'
EXAMINER = ROOT / 'sources/master3s2/master3s2.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
EXAMINER_SHA = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
PANEL = FAMILY / 'rollout_temperature_v2/development/hosted'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    raw = (PANEL/'settlement.json').read_bytes()
    settled = json.loads(raw)
    assert settled['records_reconciled'] and settled['formal_exam_games'] == 0
    indices = {(p['seed'], p['seat']): p for p in settled['archives']}
    tickets = []
    for game in sorted(settled['games'], key=lambda g: (g['seed'], g['seat'])):
        if game['arm'] != 'parent':
            continue
        seed, seat = game['seed'], game['seat']
        aliased = indices[seed, seat]
        archive = PANEL / f"{aliased['id']}.zip"
        assert aliased['sha256'] == sha(archive.read_bytes()) and game['counted']
        with zipfile.ZipFile(archive) as z:
            reg = z.read('parent/registration.json')
        r = json.loads(reg)
        assert [r['files'][k]['sha256'] for k in ('candidate','opponent')] == [PARENT_SHA, EXAMINER_SHA]
        token = f'original-action-pairs/1/{seed}/{seat}/274'.encode()
        tickets.append(dict(seed=seed, seat=seat, terrain=game['terrain'], step=274, arm='parent',
                            archive=str(archive.relative_to(FAMILY)), archive_sha256=aliased['sha256'],
                            registration_sha256=sha(reg), history_source_package=PARENT_SHA,
                            rng=int(sha(token+b'/own')[:8],16), rival_rng=int(sha(token+b'/opponent')[:8],16),
                            branch_order=['first','second'] if (seed+seat)%2==0 else ['second','first']))
    assert len(tickets)==8 and len({t['terrain'] for t in tickets})==4
    return dict(schema='real-original-action-pairs/1', source_settlement_sha256=sha(raw),
                parent=PARENT_SHA, examiner=EXAMINER_SHA, tickets=tickets,
                design='Every archived V2 seat at step 274. V2 must complete its ordinary H8 search. Compare the two top distinct final orders, including policy order when ranked in the top two. The full original chooses a common new root answer. Both arms continue with V2 versus original to official terminal. Include every skip. The stage creates new development labels on old maps, no final-test maps.')


def run(p, seed, cpu):
    os.sched_setaffinity(0, {cpu})
    os.environ['CARBON_DEBUG'] = '1'
    for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(ROOT/'code/env_src/carbon_challenge-master'), str(FAMILY/'symmetry_value/terminal_pairs')]
    import branches
    parent_receipts = load(FAMILY/'native_critic_update/public_search/response/run.py','pair_runner_loader')
    runner = parent_receipts.load_runner()
    capture = load(FAMILY/'action_rank_data/collect.py', 'pair_feature_reader')
    native_probe = load(FAMILY/'opponent_policy/search_probe/full_search_check.py', 'pair_warm_probe')
    runner.PLAN = p
    out = HERE / str(seed)
    out.mkdir(exist_ok=True)
    runner.HERE = out
    live = branches.freight.load('pair_live',branches.freight.STUDY/'cargo_tree_action_learning_dev/ladder_live/live.py')
    branches.advance = runner.checked_step
    assert sha(PARENT.read_bytes()) == PARENT_SHA and sha(EXAMINER.read_bytes()) == EXAMINER_SHA
    for ticket in (t for t in p['tickets'] if t['seed']==seed):
        seat = ticket['seat']
        key = f"{seed}_{seat}_{ticket['step']}"
        if (out/f'{key}_result.json').exists():
            continue
        mods = [branches.freight.load(f'pair_agent_{s}',PARENT if s==seat else EXAMINER) for s in (0,1)]
        for mod in mods:
            assert mod._TEMP == 1 and mod._S_ABORT == .58 and mod._SEED is None and mod._net() is not None
        active = mods[seat]
        # Warm on only the registered prior public observation, rebuild the history in the official replay below.
        helper = FAMILY/'root_rotation_policy/native/probe.py'
        method = next(node for node in ast.parse(helper.read_bytes()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'each_history')
        namespace = dict(D=FAMILY, gzip=gzip, json=json, sha=sha, zipfile=zipfile)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(helper), 'exec'), namespace)
        each_history = namespace['each_history']
        rules = json.loads((ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
        cfg = {k:v['default'] for k,v in rules['configuration'].items() if isinstance(v,dict) and 'default' in v}
        cfg['episodeSteps'] = 300
        tracker, previous = active.BeliefTracker(active._Bel, seat), None
        for obs, issued, checksum in each_history(ticket):
            if obs['step'] == ticket['step']-1:
                warm = native_probe.root_call(active,tracker,previous,obs,cfg,ticket['rng'] ^ 0x12F191AC,list(active._S_COST))
            if obs['step']==ticket['step']:
                break
            tracker.step(obs['players'],obs['step'],obs['carbon'])
            tracker.set_actions(obs['players'],issued)
            previous=obs['players']
        c = ticket | dict(own_public_sha256=checksum, initial_cost=warm['ending_costs'])
        state = runner.replay_root(c,mods,branches.freight,live)
        _, cfg, env, sim, observation, restored = state
        before = branches.freight.original_state.h([m._ST for m in mods])
        private = branches.freight.original_state.h(active._ST['rng'])
        action, detail, arrays = capture.rank_call(active,observation[seat],cfg)
        detail.update(ticket=c, restored=restored, protocol_sha256=sha(encode(p)))
        root_sha = runner.save(f'{key}_root.json.gz',detail)
        result = dict(root_sha256=root_sha, protocol_sha256=sha(encode(p)), ticket=c, status=detail['status'])
        if arrays:
            memory = io.BytesIO()
            np.savez_compressed(memory, **arrays)
            result['leaf_sha256'] = capture.publish(out/f'{key}_leaf.npz',memory.getvalue())
            orders = {arm['name']:arm['order'] for arm in detail['arms']}
            outcome = branches.continue_pair(c,orders,c['branch_order'],state,action,before,private,runner.save)
            result['outcome'] = outcome
            result['second_minus_first'] = outcome['arms']['second']['margin']-outcome['arms']['first']['margin']
        runner.save(f'{key}_result.json',result)
        print(key,detail['status'],result.get('second_minus_first'),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--register',action='store_true')
    ap.add_argument('--seed',type=int)
    ap.add_argument('--cpu',type=int)
    arg = ap.parse_args()
    p = protocol()
    data = encode(p)
    filename = HERE/'protocol.json'
    if arg.register:
        assert not filename.exists() or filename.read_bytes()==data
        filename.write_bytes(data)
        print(sha(data))
    else:
        assert filename.read_bytes()==data
        run(p,arg.seed,arg.cpu)
