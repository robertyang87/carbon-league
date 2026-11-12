"""Read-only C02 build review. No trainer, replica chain, remote, or source edits.

Run with CARBON_REPO pointing at the checkout. If unset, discover code/fastsim.py
among this script's ancestors. CARBON_REVIEW_OUT optionally saves JSON evidence;
otherwise JSON is printed to stdout. All injected inputs/output datasets are
created in a temporary directory and removed after the probe.

The fixture uses real FastCarbon observations for 3 decisions. Its replay index
says steps=300 solely to pass the existing 200-step eligibility filter; it is a
controlled parser/cache/publication fixture, not a completed arena match.
The only shell excerpt executed is replica_chain.sh's declaration gate, ending
before set_uw. Real build_top and set_uw execute; no training starts.
"""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

def find_repo():
    if os.environ.get('CARBON_REPO'):
        p = Path(os.environ['CARBON_REPO']).expanduser().resolve()
        if not (p / 'code/fastsim.py').is_file():
            raise SystemExit('CARBON_REPO must contain code/fastsim.py')
        return p
    for p in Path(__file__).resolve().parents:
        if (p / 'code/fastsim.py').is_file():
            return p
    raise SystemExit('Set CARBON_REPO to a checkout containing code/fastsim.py')

repo = find_repo()
sys.path.insert(0, str(repo / 'code'))
import numpy as np
import cnet2_build as B
from fastsim import FastCarbon
import eval_v53

# Eliminate inherited experiment switches so fixture behavior is reproducible.
env = {k: v for k, v in os.environ.items() if not k.startswith('CNET2_')}
env.update(BUILD_WORKERS='1', PYTHONDONTWRITEBYTECODE='1', CNET2_TOP_TEACHERS='review-teacher@2')
chain = (repo / 'code/replica_chain.sh').read_text()
start = chain.index('GATE=$(../.venv/bin/python -c')
end = chain.index('../.venv/bin/python set_uw.py', start)
gate_source = chain[start:end].replace('../.venv/bin/python', shlex.quote(sys.executable))
assert 'nohup' not in gate_source and 'cnet2_train' not in gate_source
report = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'tested_source_sha256': {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in ('code/cnet2_build.py', 'code/build_top.py', 'code/replica_chain.sh')},
    'scope': 'real builder and set_uw; extracted declaration gate only; synthetic parser/cache fixture; no training',
    'training_was_launched': False,
    'gate_source': gate_source,
}
checks = {}

with tempfile.TemporaryDirectory(prefix='carbon-c02-review-') as tmp:
    root = Path(tmp)
    sim = FastCarbon(eval_v53.CFG).reset(seed=9123)
    steps = []
    for t in range(4):
        if t:
            sim.step({}, {})
        steps.append([{'observation': sim.obs(p), 'status': 'ACTIVE', 'action': {}} for p in (0, 1)])
    fixture = {'steps': steps}

    def clean(x):
        if isinstance(x, str):
            return x.replace(str(root), '<TMP>')
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    def write_replay(path, value=fixture):
        with gzip.open(path, 'wt') as f:
            json.dump(value, f)

    def prepare(name):
        rd = root / name / 'corpus'
        out = root / name / 'dataset'
        (rd / 'replays').mkdir(parents=True)
        idx = []
        for i in range(2):
            file = f'replays/demo-{i}.json.gz'
            write_replay(rd / file)
            idx.append({'file': file, 'steps': 300, 'order': [0, 1],
                        'strategies': [{'name': 'review-teacher'}, {'name': 'non-teacher'}],
                        'scores': [100, 0]})
        (rd / 'index.json').write_text(json.dumps(idx))
        return rd, out

    def call(args, extra_env=None):
        p = subprocess.run(args, env=dict(env, **(extra_env or {})), capture_output=True, text=True)
        return {'exit_code': p.returncode, 'stdout': clean(p.stdout), 'stderr': clean(p.stderr)}

    def gate(out, index_count=2):
        script = 'set -u\nmark() { echo "$*"; }\n' + gate_source + '\necho GATE_PASSED_BEFORE_SET_UW\n'
        result = call(['bash', '-c', script], {'DS': str(out), 'IX': str(index_count)})
        result['passed'] = 'GATE_PASSED_BEFORE_SET_UW' in result['stdout']
        return result

    def snapshot(out):
        result = {'merged_files': sorted(p.name for p in out.glob('*.npy')),
                  'meta_exists': (out / 'meta.json').exists(),
                  'shard_names': sorted(p.name for p in (out / 'shards').glob('*.npz'))}
        if result['meta_exists']:
            result['meta'] = clean(json.loads((out / 'meta.json').read_text()))
        if (out / 'shards/_cfp.json').exists():
            result['manifest'] = json.loads((out / 'shards/_cfp.json').read_text())
        result['merged_arrays'] = {}
        for p in out.glob('*.npy'):
            arr = np.load(p)
            result['merged_arrays'][p.stem] = {
                'shape': list(arr.shape),
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        if (out / 'gid.npy').exists():
            result['gids'] = np.load(out / 'gid.npy').tolist()
        return result

    def build(name, rd, out, extra_env=None):
        result = call([sys.executable, str(repo / 'code/build_top.py'), str(out), str(rd)], extra_env)
        result['dataset'] = snapshot(out)
        result['gate'] = gate(out)
        report[name] = result
        return result

    # C02 original failure and retry, with old outputs already present.
    rd, out = prepare('original')
    r = build('healthy', rd, out)
    checks['healthy'] = r['exit_code'] == 0 and r['dataset']['meta']['valid_games'] == 2 and r['gate']['passed']
    report['healthy_set_uw'] = call([sys.executable, str(repo / 'code/set_uw.py'), str(out), '1', '1', '1'])
    report['healthy_after_set_uw'] = snapshot(out)
    (rd / 'replays/demo-0.json.gz').write_bytes(b'not a gzip archive')
    r = build('bad_gzip_default', rd, out)
    checks['default_fails_and_invalidates_product'] = r['exit_code'] == 1 and not r['dataset']['meta_exists'] and 'y.npy' not in r['dataset']['merged_files']
    checks['bad_shard_not_certified'] = 'demo-0.npz' not in r['dataset']['shard_names'] and 'demo-0.npz' not in r['dataset']['manifest']['shards']
    checks['good_shard_retained'] = 'demo-1.npz' in r['dataset']['manifest']['shards']
    r = build('bad_gzip_retry', rd, out)
    checks['failed_retry_stays_failed'] = r['exit_code'] == 1 and not r['dataset']['meta_exists']
    r = build('bad_gzip_allow_partial', rd, out, {'CNET2_ALLOW_PARTIAL': '1'})
    m = r['dataset']['meta']
    checks['partial_is_declared'] = r['exit_code'] == 0 and m['partial'] is True and m['valid_games'] == 1 and m['total'] == 3 and len(m['failed_games']) == 1 and m['failed_games'][0]['key'] == 'demo-0.json.gz' and bool(m['failed_games'][0]['error'])
    checks['partial_gate_refuses_before_set_uw'] = r['gate']['exit_code'] == 1 and not r['gate']['passed'] and 'partial dataset refused' in r['gate']['stdout']
    write_replay(rd / 'replays/demo-0.json.gz')
    r = build('restored_retry', rd, out)
    checks['restored_retry'] = r['exit_code'] == 0 and r['dataset']['meta']['valid_games'] == 2 and not r['dataset']['meta']['failed_games'] and r['gate']['passed']

    # Valid gzip/JSON with malformed observation tests processing errors, not just load.
    rd, out = prepare('processing')
    build('processing_seed', rd, out)
    malformed = copy.deepcopy(fixture)
    del malformed['steps'][0][0]['observation']['players']
    write_replay(rd / 'replays/demo-0.json.gz', malformed)
    r = build('processing_default', rd, out)
    checks['processing_failure_caught'] = r['exit_code'] == 1 and not r['dataset']['meta_exists'] and "KeyError('players')" in r['stdout']
    r = build('processing_allow_partial', rd, out, {'CNET2_ALLOW_PARTIAL': '1'})
    checks['processing_partial_declares_reason'] = r['exit_code'] == 0 and "KeyError('players')" in r['dataset']['meta']['failed_games'][0]['error'] and not r['gate']['passed']

    # Legitimate all-zero windows are distinct from failures, including cached retry.
    rd, out = prepare('zero')
    build('zero_seed', rd, out)
    for name in ('zero_window_first', 'zero_window_cached_retry'):
        r = build(name, rd, out, {'CNET2_STEP_MIN': '100'})
        m = r['dataset']['meta']
        checks[name] = r['exit_code'] == 0 and m['total'] == 0 and m['valid_games'] == 0 and m['failed_games'] == [] and m['partial'] is False and not r['dataset']['shard_names'] and not r['gate']['passed'] and 'zero-row games: demo-0.json.gz,demo-1.json.gz' in r['gate']['stdout']

    # Explicit partial with no usable games remains declared and is rejected.
    rd, out = prepare('allbad')
    for p in (rd / 'replays').glob('*.json.gz'):
        p.write_bytes(b'bad gzip')
    r = build('all_bad_default', rd, out)
    checks['all_bad_default'] = r['exit_code'] == 1 and not r['dataset']['meta_exists']
    r = build('all_bad_partial', rd, out, {'CNET2_ALLOW_PARTIAL': '1'})
    checks['all_bad_partial'] = r['exit_code'] == 0 and r['dataset']['meta']['partial'] is True and len(r['dataset']['meta']['failed_games']) == 2 and r['dataset']['meta']['total'] == 0 and not r['gate']['passed']

    # Boundary 1: a missing indexed replay is discarded by scan before err collection.
    rd, out = prepare('missing')
    build('missing_seed', rd, out)
    (rd / 'replays/demo-0.json.gz').unlink()
    r = build('missing_replay_default', rd, out)
    report['missing_replay_default']['input_index_entries'] = len(json.loads((rd / 'index.json').read_text()))
    report['missing_replay_set_uw'] = call([sys.executable, str(repo / 'code/set_uw.py'), str(out), '1', '1', '1'])
    m = r['dataset']['meta']
    checks['missing_replay_boundary_observed'] = r['exit_code'] == 0 and m['valid_games'] == 1 and m['partial'] is False and m['failed_games'] == [] and [g['key'] for g in m['games']] == ['demo-1.json.gz'] and not r['gate']['passed']

    # Boundary 2: scanner JSON exception is nonzero, but prior published files remain.
    rd, out = prepare('scanerror')
    before = build('scan_error_seed', rd, out)
    (rd / 'index.json').write_text('{bad json')
    r = build('scan_error_old_output', rd, out)
    checks['scan_error_returns_failure'] = r['exit_code'] != 0
    report['scan_error_old_output']['old_metadata_unchanged'] = r['dataset'].get('meta') == before['dataset'].get('meta')
    report['scan_error_old_output']['old_arrays_unchanged'] = r['dataset']['merged_arrays'] == before['dataset']['merged_arrays']
    report['scan_error_old_output']['gate_standalone_note'] = 'The actual chain stops at nonzero build rc before this declaration gate. Passing this isolated gate is not a launch bypass.'

    # Boundary 3: cache cert validates y row count, not presence/integrity of other fields.
    rd, out = prepare('cachemerge')
    before = build('cache_merge_seed', rd, out)
    shard = out / 'shards/demo-0.npz'
    with np.load(shard) as d:
        arrays = {k: d[k] for k in d.files if k != 'maps'}
    np.savez(shard, **arrays)
    r = build('cache_merge_old_output', rd, out)
    checks['cache_merge_failure_returns_nonzero'] = r['exit_code'] != 0
    report['cache_merge_old_output']['old_metadata_unchanged'] = r['dataset'].get('meta') == before['dataset'].get('meta')
    report['cache_merge_old_output']['merged_arrays_changed'] = r['dataset']['merged_arrays'] != before['dataset']['merged_arrays']
    report['cache_merge_old_output']['fixture_note'] = 'One cached npz field was deliberately removed while y and its certified row count remained valid. No input replay was corrupted.'
    report['cache_merge_old_output']['gate_standalone_note'] = 'Actual chain removes DS before build and stops on nonzero build rc; this is a direct reusable builder/cache boundary, not replica-chain launch bypass.'

    # Gate contract rejects missing, legacy, unreadable, and missing-count metadata.
    gd = root / 'gateonly'
    gd.mkdir()
    gate_variants = {
        'missing': None,
        'legacy': {'games': [{'n': 3}, {'n': 3}]},
        'missing_valid_count': {'games': [{'n': 3}, {'n': 3}], 'failed_games': [], 'partial': False},
        'unreadable': 'not-json',
    }
    report['gate_contract_boundaries'] = {}
    for name, data in gate_variants.items():
        p = gd / 'meta.json'
        if p.exists():
            p.unlink()
        if data is not None:
            p.write_text(data if isinstance(data, str) else json.dumps(data))
        r = gate(gd)
        report['gate_contract_boundaries'][name] = r
        checks['gate_' + name] = r['exit_code'] != 0 and not r['passed']

report['checks'] = checks
report['all_expected_checks_passed'] = all(checks.values())
result = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
print(result, end='')
if os.environ.get('CARBON_REVIEW_OUT'):
    out = Path(os.environ['CARBON_REVIEW_OUT']).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'c02-results.json').write_text(result)
if not report['all_expected_checks_passed']:
    raise SystemExit(1)
