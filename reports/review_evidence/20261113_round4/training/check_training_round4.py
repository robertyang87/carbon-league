"""Read-only source review and tiny local input probes; never execute cloud scripts or train.
CARBON_REPO chooses checkout; CARBON_REVIEW_OUT chooses artifact directory.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as Fn

torch.set_num_threads(1)
if os.environ.get('CARBON_REPO'):
    REPO = Path(os.environ['CARBON_REPO']).resolve()
else:
    REPO = next((p for p in Path(__file__).resolve().parents if (p/'code/fastsim.py').exists()), None)
    if REPO is None:
        raise SystemExit('Set CARBON_REPO to the checkout root.')
OUT = Path(os.environ.get('CARBON_REVIEW_OUT', Path(__file__).parent)).resolve()
OUT.mkdir(parents=True, exist_ok=True)
BASE = '14d53320b20d58bdb82676f7fa866b8e11a55062'
DESIGN = '8c615a912430a0e5355c2ece5a166aa458e23cb5'
def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()

def run_python(path, args, env=None):
    e = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if env:
        e.update(env)
    p = subprocess.run([sys.executable, str(path), *map(str, args)], env=e, text=True,
                       capture_output=True, timeout=30)
    return {'exit': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}

result = {'head': git('rev-parse', 'HEAD'), 'baseline': BASE, 'design_baseline': DESIGN,
          'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
          'scope': 'No cloud script executed; no training, game rollout, remote operation, or source write.'}
core = ['rl2_ppo_train.py', 'rl2_ppo_rollout.py', 'rl2_ppo_adapters.py', 'rl2_ppo_cnet2.py',
        'rl2_ppo_server.py', 'rl2_actor.py', 'rl2_common.py', 'fastsim.py', 'cnet2_feat.py',
        'cnet2_model.py', 'cnet2_np.py', 'cnet2_train.py']
result['core_blobs'] = []
for name in core:
    f = 'code/' + name
    blobs = {r: git('rev-parse', r + ':' + f) for r in (BASE, DESIGN, 'HEAD')}
    result['core_blobs'].append({'path': f, 'blobs': blobs, 'identical': len(set(blobs.values())) == 1})
result['all_core_identical'] = all(x['identical'] for x in result['core_blobs'])

changed = git('diff', '--name-only', BASE + '..HEAD', '--', 'code/cloud').splitlines()
all_cloud = git('ls-tree', '-r', '--name-only', 'HEAD', 'code/cloud').splitlines()
seed_scripts = []
fixed_cases = []
for f in all_cloud:
    txt = (REPO / f).read_text()
    if 'EVO_SEEDS' not in txt:
        continue
    lines = txt.splitlines()
    seeds_at = []
    for i, line in enumerate(lines, 1):
        match = re.search(r'EVO_SEEDS=([0-9,]+)(?:\s|$)', line)
        if match:
            seedstr = match.group(1)
            seedlist = [int(x) for x in seedstr.split(',')]
            item = {'path': f, 'line': i, 'seed_count': len(seedlist), 'unique_seeds': len(set(seedlist)),
                    'n_is_zero': ' 0 0 ' in line, 'seeds': seedlist}
            item['probe'] = run_python(REPO / 'code/bigeval.py',
                                      ['not_loaded_a.py', 'not_loaded_b.py', 0, 0, 'review_seed_contract'],
                                      {'EVO_SEEDS': seedstr})
            fixed_cases.append(item)
        if 'EVO_SEEDS' in line:
            seeds_at.append(i)
    seed_scripts.append({'path': f, 'new_since_report': f in changed, 'seed_assignment_lines': seeds_at,
                         'completion_lines': [{'line': i, 'text': l.strip()} for i, l in enumerate(lines, 1)
                                              if any(x in l for x in ('SX_DONE', 'CB_DONE', 'S50_DONE'))]})
result['all_cloud_scripts_with_EVO_SEEDS'] = seed_scripts
result['literal_seed_calls'] = fixed_cases
result['resume_100_missing_case'] = run_python(REPO / 'code/bigeval.py',
    ['not_loaded_a.py', 'not_loaded_b.py', 0, 0, 'review_resume_contract'],
    {'EVO_SEEDS': ','.join(str(s) for s in range(994000, 994100))})
# Independent bash language microcase, not a repository/cloud script. No files or jobs beyond an exit-2 subshell.
p = subprocess.run(['bash', '-c', '(exit 2) & wait; printf "WAIT_RC=%s\\nCB_DONE\\n" "$?"'],
                   text=True, capture_output=True, check=False, timeout=5)
result['bash_wait_semantics_only'] = {'command': '(exit 2) & wait; printf WAIT_RC_and_CB_DONE',
                                     'exit': p.returncode, 'stdout': p.stdout.strip(),
                                     'note': 'No real completion marker was written.'}

with tempfile.TemporaryDirectory(prefix='carbon-training-r4-') as td:
    root = Path(td)
    def make_ds(name, weights):
        d = root / name; d.mkdir()
        n = len(weights)
        kind = np.full((n,24), 5, np.int16); kind[:,0] = 0; kind[:,1] = 1; kind[:,10:12] = 2
        mask = np.zeros((n,24), bool); mask[:,[0,1,10,11]] = True
        uw = np.zeros((n,12), np.float32); uw[:,[1,10,11]] = 1
        y = np.full((n,12), -1, np.int16); y[:,[0,1,10,11]] = 0
        for key, val in {'w': np.array(weights,np.float32), 'uw': uw, 'tok_kind': kind,
                         'tok_mask': mask, 'y': y}.items(): np.save(d/(key+'.npy'), val)
        return d
    normal = make_ds('normal', [.5, 2, 0]); target = root/'anchor'
    nr = run_python(REPO/'code/awr_collector_anchor.py', [normal, target])
    w = np.load(target/'w.npy'); uw = np.load(target/'uw.npy')
    nr.update(collector_effective_weight=(w*uw[:,0]).tolist(), zero_weight_stays_zero=bool(uw[2,0] == 0),
              source_uw_unchanged=bool(np.all(np.load(normal/'uw.npy')[:,0] == 0)),
              expected_positive_row_weight=float(w.mean()))
    result['anchor_normal'] = nr
    before_m3s = np.full((3,12), .4, np.float32)
    np.save(normal/'uw.npy', before_m3s)
    mr = run_python(REPO/'code/m3s_uw.py', [normal])
    new = np.load(normal/'uw.npy'); bak = np.load(normal/'uw_build.npy')
    mr.update(collector_zero=bool(np.all(new[:,0] == 0)), planter_base_one=bool(np.all(new[:,[1,10,11]] == 1)),
              padding_zero=bool(np.all(new[:,2:10] == 0)), backup_exists=True,
              backup_matches_prior=bool(np.array_equal(bak, before_m3s)))
    result['m3s_normal'] = mr
    edge = make_ds('small_positive_weight', [1e-6, 1]); edgeout = root/'edge_anchor'
    er = run_python(REPO/'code/awr_collector_anchor.py', [edge, edgeout])
    w = torch.from_numpy(np.load(edgeout/'w.npy')); uw = torch.from_numpy(np.load(edgeout/'uw.npy'))
    # Extract the repository's real cached batch_loss without invoking main/train_cached.
    tree = ast.parse((REPO/'code/cnet2_train.py').read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_cached')
    fn = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'batch_loss')
    class FixedNet:
        def ctx_from(self, pre, teacher): return pre
        def decode(self, b, tok, ctx): return torch.zeros((2,12,5))
    ns = {'torch': torch, 'Fn': Fn, 'net': FixedNet(), 'REF': None}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(REPO/'code/cnet2_train.py'), 'exec'), ns)
    b = {'w': w.half().float(), 'uw': uw.half().float(), 'y': torch.from_numpy(np.load(edgeout/'y.npy')).long(),
         'teacher': torch.zeros(2,dtype=torch.long)}
    loss,_ = ns['batch_loss'](b, None, None)
    er.update(raw_uw_finite=bool(torch.isfinite(uw).all()), raw_collector_uw=uw[:,0].tolist(),
              cached_uw_finite=bool(torch.isfinite(b['uw']).all()),
              cached_collector_uw=[str(x) for x in b['uw'][:,0].tolist()],
              real_cached_loss_finite=bool(torch.isfinite(loss)), real_cached_loss=str(loss.item()),
              limitation='Constructed valid float32 weights, default SCALE=1; no real dataset extrema or failure prevalence measured.')
    result['anchor_float16_boundary'] = er

result['repo_status_after'] = git('status', '--short')
(OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'result': str(OUT/'results.json'), 'all_core_identical': result['all_core_identical'],
                  'affected_seed_scripts': [x['path'] for x in seed_scripts],
                  'literal_call_counts': [x['seed_count'] for x in fixed_cases],
                  'literal_call_exits': [x['probe']['exit'] for x in fixed_cases],
                  'wait': result['bash_wait_semantics_only']['stdout'],
                  'anchor_edge_loss': result['anchor_float16_boundary']['real_cached_loss']}, ensure_ascii=False))
