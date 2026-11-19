"""Check the standalone entry point in an isolated offline process on every sealed tail."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import types

HERE = Path(__file__).resolve().parent


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def clean(orders):
    return {uid: cmd for uid, cmd in orders.items() if cmd and cmd != 'NONE'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def normalized(audit):
    return {k:v for k,v in audit.items() if k != 'cpu_ms'}


def worker():
    def no_network(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('network disabled during package check')
    sys.addaudithook(no_network)
    require(sys.flags.isolated and not any('sapling-return-search' in p for p in sys.path), 'not isolated')
    sentinel = types.ModuleType('fastsim')
    sys.modules['fastsim'] = sentinel
    path = Path('candidate.py')
    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location('standalone_candidate', path)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    import_ms = (time.perf_counter()-start)*1000
    result = []

    def checked(obs, cfg, fact, expected, audit):
        t = time.perf_counter()
        orders = clean(pack.agent(obs,cfg))
        ms = 1000*(time.perf_counter()-t)
        turn, seat, base, trace = pack._QUEUE_TRACE
        require((turn,seat)==(obs['step'],obs['player']) and clean(base)==clean(fact), 'genuine v1 order mismatch')
        require(orders==clean(expected), f"candidate order mismatch at {turn}/{seat}")
        if audit is None:
            require(trace is None, 'outside-window decision')
        else:
            require(normalized(audit)==normalized(trace), f'policy audit mismatch at {turn}/{seat}')
        return {'t':turn, 'v1':clean(base), 'emitted':orders, 'wall_ms':ms, 'policy_cpu_ms': trace['cpu_ms'] if trace else None}

    for game in json.load(sys.stdin):
        seat, cfg, actions = game['seat'], game['cfg'], game['actions']
        sim = pack._QUEUE_SIM.FastCarbon(cfg).reset(full_carbon=game['initial'])
        trial = {'seed':game['seed'], 'seat':seat, 'raw_sha256':game['raw_sha256'], 'tail':[]}
        for t, joint in enumerate(actions):
            obs = sim.obs(seat)
            if t == 0:
                trial['opening_wall_ms'] = checked(obs,cfg,joint[seat],joint[seat],None)['wall_ms']
            elif t < 297:
                pack._FEAT.observe(obs)
                pack._queue_remember(obs)
            else:
                target = game['roots'][str(t)]
                call = checked(obs,cfg,joint[seat],target['candidate_own'],target['audit'])
                trial['tail'].append(call)
                if t == 297 and game.get('fork'):
                    root = sim.get_state()
                    history = copy.deepcopy(pack._FEAT), copy.deepcopy(pack._QUEUE_PREVIOUS)
                    changed = list(joint)
                    changed[seat] = call['emitted']
                    sim.step(*changed)
                    fork = game['fork']
                    require(sim.obs(seat)==fork['obs'], 'fork observation differs from independently evaluated state')
                    trial['fork_last_call'] = checked(sim.obs(seat), cfg, fork['v1'], fork['selected'], fork['audit'])
                    trial['fork_last_call']['differs_from_taped_candidate'] = trial['fork_last_call']['emitted'] != clean(game['roots']['298']['candidate_own'])
                    sim.set_state(root)
                    pack._FEAT, pack._QUEUE_PREVIOUS = history
            sim.step(*joint)
        require(list(sim.cash)==game['terminal_cash'], 'factual terminal cash drift')
        result.append(trial)
    require(sys.modules['fastsim'] is sentinel, 'candidate polluted global fastsim')
    print(json.dumps({'import_ms':import_ms, 'games':result, 'affinity':sorted(os.sched_getaffinity(0)),
                      'niceness':os.getpriority(os.PRIO_PROCESS,0), 'fastsim_sentinel_preserved':True}))


def main():
    from build import build
    from study import ROOT, FastCarbon, protocol, write
    import policy
    register = protocol()
    pkg = load(HERE/'package.json')
    replay = load(HERE/'replay_roots.json.gz')
    detail = load(HERE/'root_audit.json.gz')
    verification = load(HERE/'study_results.json')
    sensitivity = load(HERE/'v1_last_reply.json.gz')
    require(pkg['candidate_sha256'] == sha(HERE/'candidate.py') and pkg['parent_sha256']==register['source_parent'], 'package identity')
    for name, checksum in verification['bindings'].items():
        if name in ('policy.py', 'replay_roots.json.gz', 'root_audit.json.gz'):
            require(sha(HERE/name)==checksum, 'study binding')
    require(sensitivity['offered_actions_sha256']==sha(HERE/'root_audit.json.gz') and
            sensitivity['protocol_sha256']==sha(HERE/'protocol.json') and sensitivity['original_pack_sha256']==pkg['parent_sha256'], 'v1 sensitivity binding')
    rows = {(r['seed'],r['seat'],r['t']):r for r in detail}
    sources = {}
    for batch in register['source_batches']:
        receipt = ROOT/batch['receipt']
        require(sha(receipt)==batch['receipt_sha256'], 'receipt drift')
        sources.update({(g['seed'],g['seat']):g for g in load(receipt)['games']})
    request = []
    for g in replay['games']:
        seed, seat = g['seed'], g['seat']
        ref = sources[seed,seat]
        path = ROOT/ref['raw_path']
        require(sha(path) == ref['raw_sha256'] == g['raw_sha256'], 'raw mismatch')
        raw = load(path)
        require(g['steps']==299 and not g['excluded'], 'missing registered tail')
        item = dict(seed=seed,seat=seat,initial=raw['initial_full_carbon'],cfg=raw['game_configuration'],
                    actions=raw['actions'],raw_sha256=g['raw_sha256'],terminal_cash=g['original']['cash'],
                    roots={str(t):{n: rows[seed,seat,t][n] for n in ('candidate_own','audit')} for t in (297,298)})
        if rows[seed,seat,297]['changed']:
            comparison = next(x for x in sensitivity['roots'] if (x['seed'],x['seat'])==(seed,seat))
            branch = next(a for a in comparison['arms'] if a['arm']=='candidate')
            joint = list(raw['actions'][297]); joint[seat] = item['roots']['297']['candidate_own']
            sim = FastCarbon(g['config']).set_state(g['states']['297']); sim.step(*joint)
            public = sim.obs(seat)
            own = branch['v1_last_commands']
            selected, audit = policy.choose(public, {297:g['history']['297'][seat]}, own, g['config'])
            item['fork'] = {'v1':own, 'selected':selected, 'audit':audit, 'obs':public}
        request.append(item)
    require(len(rows)==56 and len(request)==28, 'wrong allocation')
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        rebuilt = build(tmp)
        require(rebuilt==pkg and sha(tmp/'candidate.py')==pkg['candidate_sha256'], 'non-reproducible build')
        (tmp/'package.json').unlink()
        shutil.copyfile(__file__,tmp/'check.py')
        env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        run = subprocess.run([sys.executable,'-I','-B',str(tmp/'check.py'),'--worker'], input=json.dumps(request),
                             text=True, capture_output=True, env=env, cwd=tmp)
        require(run.returncode==0, run.stderr[-3000:])
        audited = json.loads(run.stdout)
    roots = [c for g in audited['games'] for c in g['tail']]
    forks = [g['fork_last_call'] for g in audited['games'] if 'fork_last_call' in g]
    require(len(roots)==56 and len(forks)==sum(r['changed'] for r in detail)==8, 'incomplete package evidence')
    def distribution(values):
        return {'n':len(values), 'min':min(values), 'median':statistics.median(values), 'max':max(values)}
    audited.update(candidate_sha256=pkg['candidate_sha256'], package_manifest_sha256=sha(HERE/'package.json'),
                   verification_source_sha256=sha(Path(__file__)), source_audit_sha256=sha(HERE/'root_audit.json.gz'),
                   independent_parent_replies_sha256=sha(HERE/'v1_last_reply.json.gz'),
                   factual_tail_matches=len(roots), factual_tail_edits=sum(c['v1']!=c['emitted'] for c in roots),
                   fork_last_matches=len(forks), fork_last_changes_to_recomputed_v1=sum(c['v1']!=c['emitted'] for c in forks),
                   fork_last_different_from_factual=sum(c['differs_from_taped_candidate'] for c in forks),
                   terminal_entry_wall_ms=distribution([c['wall_ms'] for c in roots]),
                   opening_entry_wall_ms=distribution([g['opening_wall_ms'] for g in audited['games']]),
                   timing_scope='One isolated process; 28 opening and 56 recorded-tail full entry calls plus 8 changed-branch calls. Intermediate turns advanced real v1 feature history and wrapper rosters; no full deployment deadline evaluated.')
    write(HERE/'package_parity.json.gz',audited)
    print(json.dumps({k:v for k,v in audited.items() if k!='games'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    if sys.argv[1:] == ['--worker']:
        worker()
    else:
        main()
